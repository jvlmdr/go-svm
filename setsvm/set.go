package setsvm

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
	if len(set) == 0 {
		panic("empty")
	}
	var n int
	for i, xi := range set {
		if i == 0 {
			n = len(xi)
			continue
		}
		if n != len(xi) {
			panic(fmt.Sprintf("vector dims: found %d and %d", n, len(xi)))
		}
	}
	return n
}

func (set Slice) At(i int) []float64 {
	return set[i]
}
