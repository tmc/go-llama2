package main

/*
#include <math.h>

float sqrt32(float x) {
    return sqrtf(x);
}
*/
import "C"
import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
)

type Config struct {
	Dim       int32 // transformer dimension
	HiddenDim int32 // for ffn layers
	NLayers   int32 // number of layers
	NHeads    int32 // number of query heads
	NKvHeads  int32 // number of key/value heads (can be < query heads because of multiquery)
	VocabSize int32 // vocabulary size, usually 256 (byte-level)
	SeqLen    int32 // max sequence length
}

type TransformerWeights struct {
	TokenEmbeddingTable []float32 // (vocab_size, dim)
	RmsAttWeight        []float32 // (layer, dim) rmsnorm weights
	RmsFfnWeight        []float32 // (layer, dim)
	Wq                  []float32 // (layer, dim, dim)
	Wk                  []float32 // (layer, dim, dim)
	Wv                  []float32 // (layer, dim, dim)
	Wo                  []float32 // (layer, dim, dim)
	W1                  []float32 // (layer, hidden_dim, dim)
	W2                  []float32 // (layer, dim, hidden_dim)
	W3                  []float32 // (layer, hidden_dim, dim)
	RmsFinalWeight      []float32 // (dim,)
	FreqCisReal         []float32 // (seq_len, dim/2)
	FreqCisImag         []float32 // (seq_len, dim/2)
}

type RunState struct {
	X          []float32 // activation at current time stamp (dim,)
	Xb         []float32 // same, but inside a residual branch (dim,)
	Xb2        []float32 // an additional buffer just for convenience (dim,)
	Hb         []float32 // buffer for hidden dimension in the ffn (hidden_dim,)
	Hb2        []float32 // buffer for hidden dimension in the ffn (hidden_dim,)
	Q          []float32 // query (dim,)
	K          []float32 // key (dim,)
	V          []float32 // value (dim,)
	Att        []float32 // buffer for scores/attention values (seq_len,)
	Logits     []float32 // output logits
	KeyCache   []float32 // (layer, seq_len, dim)
	ValueCache []float32 // (layer, seq_len, dim)
}

func main() {
	// disable stdout buffering
	// no equivalent in Go, stdout is unbuffered by default

	// poor man's Go flag.Parse()
	checkpoint := ""
	temperature := 0.9
	steps := int32(256)
	// 'checkpoint' is necessary arg
	if len(os.Args) < 2 {
		fmt.Printf("Usage: %s <checkpoint_file> [temperature] [steps]\n", os.Args[0])
		os.Exit(1)
	}
	checkpoint = os.Args[1]
	// temperature is optional
	if len(os.Args) >= 3 {
		temperature, _ = strconv.ParseFloat(os.Args[2], 64)
	}
	// seed is optional
	if len(os.Args) >= 4 {
		s, _ := strconv.Atoi(os.Args[3])
		steps = int32(s)
	}

	// read in the config header
	var config Config
	file, err := os.Open(checkpoint)
	if err != nil {
		fmt.Println("Unable to open file!")
		os.Exit(1)
	}
	err = binary.Read(file, binary.LittleEndian, &config)
	if err != nil {
		fmt.Println("binary.Read failed:", err)
		os.Exit(1)
	}

	// create and init the Transformer
	var weights TransformerWeights
	allocWeights(&weights, &config)
	checkpointInitWeights(&weights, &config, file)
	file.Close()

	// create and init the application RunState
	var state RunState
	allocRunState(&state, &config)
	/*

	   // right now we cannot run for more than config.seq_len steps
	   if (steps <= 0 || steps > config.seq_len) { steps = config.seq_len; }
	*/

	if steps <= 0 || steps > config.SeqLen {
		steps = config.SeqLen
	}

	vocab := make([]string, config.VocabSize)
	tokfile, err := os.Open("../tokenizer.bin")
	if err != nil {
		log.Fatalln("Unable to open tokenizer.bin", err)
	}
	defer tokfile.Close()
	for i := int32(0); i < config.VocabSize; i++ {
		var length int32
		err = binary.Read(tokfile, binary.LittleEndian, &length)
		if err != nil {
			log.Fatalln("Unable to read length from tokenizer.bin", err)
		}
		bytes := make([]byte, length)
		_, err = io.ReadFull(tokfile, bytes)
		if err != nil {
			log.Fatalln("Unable to read bytes from tokenizer.bin", err)
		}
		vocab[i] = string(bytes)
	}

	// the current position we are in
	start := time.Now()
	var next, token, pos int32
	token = 1 // 1 = BOS token in Llama-2 sentencepiece
	pos = 0

	for pos < steps {
		// forward the transformer to get logits for the next token
		transformer(token, pos, &config, &state, &weights)

		// sample the next token
		if temperature == 0.0 {
			// greedy argmax sampling
			next = argmax(state.Logits)
		} else {
			// apply the temperature to the logits
			for q := int32(0); q < config.VocabSize; q++ {
				state.Logits[q] /= float32(temperature)
			}
			// apply softmax to the logits to get the probabilities for next token
			softmax(state.Logits)
			// we now want to sample from this distribution to get the next token
			next = sample(state.Logits)
		}
		fmt.Print(vocab[next])

		// advance forward
		token = next
		pos++
	}

	// report achieved tok/s
	end := time.Now()
	fmt.Printf("\nachieved tok/s: %f\n", float64(steps)/end.Sub(start).Seconds())
}

func allocWeights(w *TransformerWeights, p *Config) {
	dim := p.Dim
	hiddenDim := p.HiddenDim
	nLayers := p.NLayers
	vocabSize := p.VocabSize
	seqLen := p.SeqLen
	dim2 := dim * dim
	hiddenDimDim := hiddenDim * dim

	w.TokenEmbeddingTable = make([]float32, vocabSize*dim)
	w.RmsAttWeight = make([]float32, nLayers*dim)
	w.RmsFfnWeight = make([]float32, nLayers*dim)
	w.Wq = make([]float32, nLayers*dim2)
	w.Wk = make([]float32, nLayers*dim2)
	w.Wv = make([]float32, nLayers*dim2)
	w.Wo = make([]float32, nLayers*dim2)
	w.W1 = make([]float32, nLayers*hiddenDimDim)
	w.W2 = make([]float32, nLayers*dim*hiddenDim)
	w.W3 = make([]float32, nLayers*hiddenDimDim)
	w.RmsFinalWeight = make([]float32, dim)
	w.FreqCisReal = make([]float32, seqLen*dim/2)
	w.FreqCisImag = make([]float32, seqLen*dim/2)
}

func checkpointInitWeights(w *TransformerWeights, p *Config, file io.Reader) {
	binary.Read(file, binary.LittleEndian, &w.TokenEmbeddingTable)
	binary.Read(file, binary.LittleEndian, &w.RmsAttWeight)
	binary.Read(file, binary.LittleEndian, &w.Wq)
	binary.Read(file, binary.LittleEndian, &w.Wk)
	binary.Read(file, binary.LittleEndian, &w.Wv)
	binary.Read(file, binary.LittleEndian, &w.Wo)
	binary.Read(file, binary.LittleEndian, &w.RmsFfnWeight)
	binary.Read(file, binary.LittleEndian, &w.W1)
	binary.Read(file, binary.LittleEndian, &w.W2)
	binary.Read(file, binary.LittleEndian, &w.W3)
	binary.Read(file, binary.LittleEndian, &w.RmsFinalWeight)
	binary.Read(file, binary.LittleEndian, &w.FreqCisReal)
	binary.Read(file, binary.LittleEndian, &w.FreqCisImag)
}

func allocRunState(s *RunState, p *Config) {
	dim := p.Dim
	hiddenDim := p.HiddenDim
	nLayers := p.NLayers
	vocabSize := p.VocabSize
	seqLen := p.SeqLen

	s.X = make([]float32, dim)
	s.Xb = make([]float32, dim)
	s.Xb2 = make([]float32, dim)
	s.Hb = make([]float32, hiddenDim)
	s.Hb2 = make([]float32, hiddenDim)
	s.Q = make([]float32, dim)
	s.K = make([]float32, dim)
	s.V = make([]float32, dim)
	s.Att = make([]float32, p.NHeads*seqLen)
	s.Logits = make([]float32, vocabSize)
	s.KeyCache = make([]float32, nLayers*seqLen*dim)
	s.ValueCache = make([]float32, nLayers*seqLen*dim)
}

func softmax(x []float32) {
	size := len(x)
	if size == 1 {
		x[0] = 1.0
		return
	}

	// find max value (for numerical stability)
	maxVal := x[0]
	for i := 1; i < size; i++ {
		if x[i] > maxVal {
			maxVal = x[i]
		}
	}

	// e^x
	for i := 0; i < size; i++ {
		x[i] = float32(math.Exp(float64(x[i] - maxVal)))
	}

	// normalize
	sum := float32(0.0)
	for i := 0; i < size; i++ {
		sum += x[i]
	}
	for i := 0; i < size; i++ {
		x[i] /= sum
	}
}

func matmul(xout, x, w []float32) {
	// W (d,n) @ x (n,) -> xout (d,)
	n := len(x)
	d := len(w) / n
	for i := 0; i < d; i++ {
		val := float32(0)
		for j := 0; j < n; j++ {
			val += w[i*n+j] * x[j]
		}
		xout[i] = val
	}
}

func transformer(token int32, pos int32, p *Config, s *RunState, w *TransformerWeights) {
	// a few convenience variables
	x := s.X
	dim := p.Dim
	hiddenDim := p.HiddenDim
	headSize := dim / p.NHeads

	// copy the token embedding into x
	contentRow := w.TokenEmbeddingTable[token*dim : (token+1)*dim]
	copy(x, contentRow)

	// pluck out the "pos" row of freqCisReal and freqCisImag

	freqCisRealRow := w.FreqCisReal[pos*headSize/2 : (pos+1)*headSize/2]
	freqCisImagRow := w.FreqCisImag[pos*headSize/2 : (pos+1)*headSize/2]

	// forward all the layers
	xSum := float32(0.0)
	_ = xSum
	for l := int32(0); l < p.NLayers; l++ {
		// attention rmsnorm
		rmsNorm(s.Xb, x, w.RmsAttWeight[l*dim:(l+1)*dim])

		// qkv matmuls for this position
		matmul(s.Q, s.Xb, w.Wq[l*dim*dim:(l+1)*dim*dim])
		matmul(s.K, s.Xb, w.Wk[l*dim*dim:(l+1)*dim*dim])
		matmul(s.V, s.Xb, w.Wv[l*dim*dim:(l+1)*dim*dim])

		// apply RoPE rotation to the q and k vectors for each head
		for h := int32(0); h < p.NHeads; h++ {
			// get the q and k vectors for this head
			q := s.Q[h*headSize : (h+1)*headSize]
			k := s.K[h*headSize : (h+1)*headSize]

			// rotate q and k by the freqCisReal and freqCisImag
			for i := int32(0); i < headSize; i += 2 {
				q0 := q[i]
				q1 := q[i+1]
				k0 := k[i]
				k1 := k[i+1]
				fcr := freqCisRealRow[i/2]
				fci := freqCisImagRow[i/2]
				q[i] = q0*fcr - q1*fci
				q[i+1] = q0*fci + q1*fcr
				k[i] = k0*fcr - k1*fci
				k[i+1] = k0*fci + k1*fcr
			}
		}

		// save key,value at this time step (pos) to our kv cache
		loff := l * p.SeqLen * dim // kv cache layer offset for convenience
		keyCacheRow := s.KeyCache[loff+pos*dim : loff+(pos+1)*dim]
		valueCacheRow := s.ValueCache[loff+pos*dim : loff+(pos+1)*dim]
		copy(keyCacheRow, s.K)
		copy(valueCacheRow, s.V)

		// multihead attention. iterate over all heads
		for h := int32(0); h < p.NHeads; h++ {
			// get the query vector for this head
			q := s.Q[h*headSize : (h+1)*headSize]
			// iterate over all timesteps, including the current one
			for t := int32(0); t <= pos; t++ {
				// get the key vector for this head and at this timestep
				k := s.KeyCache[loff+t*dim+h*headSize : loff+t*dim+(h+1)*headSize]
				// calculate the attention score as the dot product of q and k
				score := float32(0.0)
				for i := int32(0); i < headSize; i++ {
					score += q[i] * k[i]
				}
				score = score / float32(math.Sqrt(float64(headSize)))
				// save the score to the attention buffer
			}

			// softmax the scores to get attention weights, from 0..pos inclusively
			softmax(s.Att[:pos+1])

			// weighted sum of the values, store back into xb
			for i := int32(0); i < headSize; i++ {
				val := 0.0
				for t := int32(0); t <= pos; t++ {
					val += float64(s.Att[t] * s.ValueCache[loff+t*dim+h*headSize+i]) // note bad locality
				}
				s.Xb[h*headSize+i] = float32(val)
			}
		}

		// final matmul to get the output of the attention
		matmul(s.Xb2, s.Xb, w.Wo[l*dim*dim:(l+1)*dim*dim])

		// residual connection back into x
		accum(x, s.Xb2)

		// ffn rmsnorm
		rmsNorm(s.Xb, x, w.RmsFfnWeight[l*dim:(l+1)*dim])

		// Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
		// first calculate self.w1(x) and self.w3(x)
		matmul(s.Hb, s.Xb, w.W1[l*dim*hiddenDim:(l+1)*dim*hiddenDim])
		matmul(s.Hb2, s.Xb, w.W3[l*dim*hiddenDim:(l+1)*dim*hiddenDim])

		// F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
		for i := int32(0); i < hiddenDim; i++ {
			s.Hb[i] = s.Hb[i] * (1.0 / (1.0 + float32(math.Exp(-float64(s.Hb[i])))))
		}

		// elementwise multiply with w3(x)
		for i := int32(0); i < hiddenDim; i++ {
			s.Hb[i] = s.Hb[i] * s.Hb2[i]
		}

		// final matmul to get the output of the ffn
		matmul(s.Xb, s.Hb, w.W2[l*dim*hiddenDim:(l+1)*dim*hiddenDim])

		// residual connection
		accum(x, s.Xb)
	}

	// final rmsnorm
	rmsNorm(x, x, w.RmsFinalWeight)

	// classifier into logits
	matmul(s.Logits, x, w.TokenEmbeddingTable)
}

func accum(a, b []float32) {
	for i := range a {
		a[i] += b[i]
	}

}

func sqrt(x float32) float32 {
	return float32(C.sqrt32(C.float(x)))
}

func rmsNorm(dest, src, weight []float32) {
	sumSquares := float32(0.0)
	for _, x := range src {
		sumSquares += x * x
	}
	// fmt.Printf("rmsnorm ss0: %.20f\n", sumSquares)
	ss := sumSquares/float32(len(src)) + float32(1e-5)
	// fmt.Printf("rmsnorm ss1: %.20f\n", ss)
	ss = 1.0 / sqrt(ss)
	// fmt.Printf("rmsnorm ss2: %.20f\n", ss)
	for i, x := range src {
		dest[i] = weight[i] * (ss * x)
		// fmt.Printf("rmsnorm i: %d x: %.20f ss: %.20f weight: %.20f dest: %.20f\n", i, x, ss, weight[i], dest[i])
	}
}

func argmax(v []float32) int32 {
	// return argmax of v
	maxI := 0
	maxP := v[0]
	// fmt.Printf("new max %d %f\n", maxI, maxP)
	for i := 1; i < len(v); i++ {
		if v[i] > maxP {
			maxI = i
			maxP = v[i]
			// print:
			// fmt.Printf("new max %d %f\n", maxI, maxP)
		}
	}
	return int32(maxI)
}

func sample(probabilities []float32) int32 {
	// sample index from probabilities, they must sum to 1
	r := rand.Float32()
	cdf := float32(0.0)
	for i, probability := range probabilities {
		cdf += probability
		if r < cdf {
			return int32(i)
		}
	}
	return int32(len(probabilities)) - 1 // in case of rounding errors
}
