package svm

import "fmt"

type Set interface {
	Len() int
	Dim() int
	At(int) []float64
}

type Slice [][]float64

func (set Slice) Len() int {
	return len(set)
}

func (set Slice) Dim() int {
	return len(set[0])
}

func (set Slice) At(i int) []float64 {
	return set[i]
}

// Dimension returns the dimension of
// the vectors in a set of examples.
// Returns an error if vectors of different dimension are found
// or there are no vectors.
func dimension(x [][]float64) (int, error) {
	if len(x) == 0 {
		return 0, fmt.Errorf("empty list")
	}
	n := len(x[0])
	for _, xi := range x {
		if n != len(xi) {
			return 0, fmt.Errorf("vector dims: found %d and %d", n, len(xi))
		}
	}
	return n, nil
}
