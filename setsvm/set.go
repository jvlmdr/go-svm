package setsvm

import "fmt"

type Set interface {
	NumSets() int
	NumExamples() int
	Dim() int
	SetLen(i int) int
	At(i, j int) []float64
}

type Slice [][][]float64

func (set Slice) NumSets() int {
	return len(set)
}

func (set Slice) NumExamples() int {
	var n int
	for i := range set {
		n += len(set[i])
	}
	return n
}

func (set Slice) SetLen(i int) int {
	return len(set[i])
}

func (set Slice) Dim() int {
	m, err := dimension(set)
	if err != nil {
		panic(err)
	}
	return m
}

func (set Slice) At(i, j int) []float64 {
	return set[i][j]
}

// Dimension returns the dimension of
// the vectors in a set of example sets.
// Returns an error if vectors of different dimension are found
// or there are no vectors.
func dimension(x [][][]float64) (int, error) {
	n := -1
	for i := range x {
		for j := range x[i] {
			if n < 0 {
				n = len(x[i][j])
				continue
			}
			if n != len(x[i][j]) {
				return 0, fmt.Errorf("vector dims: found %d and %d", n, len(x[i][j]))
			}
		}
	}
	if n < 0 {
		return 0, fmt.Errorf("no vectors found")
	}
	return n, nil
}
