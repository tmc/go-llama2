package main

import (
	"encoding/binary"
	"os"
	"testing"
)

func BenchmarkTransformer(b *testing.B) {
	// Define a table of test cases
	cases := []struct {
		name      string
		token     int32
		pos       int32
		modelPath string
	}{
		{"basic", 0, 0, "../stories15M.bin"},
	}

	// Loop over test cases
	for _, tc := range cases {
		// Capture the test case for the parallel test
		tc := tc
		config := &Config{}
		file, err := os.Open(tc.modelPath)
		if err != nil {
			b.Fatal(err)
		}
		err = binary.Read(file, binary.LittleEndian, config)
		if err != nil {
			b.Fatal(err)
		}
		state := &RunState{}
		allocRunState(state, config)
		weights := &TransformerWeights{}
		allocWeights(weights, config)
		checkpointInitWeights(weights, config, file)
		// Run each test case as a sub-benchmark
		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer() // Reset timer as setup time should not be included in the benchmark
			for i := 0; i < b.N; i++ {
				transformer(tc.token, tc.pos, config, state, weights)
			}
		})
	}
}
