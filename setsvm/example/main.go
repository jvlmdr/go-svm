package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/jvlmdr/go-svm/setsvm"
)

func main() {
	var (
		numSets1 = 1000
		numSets2 = 5000
		setLen1  = 500
		setLen2  = 100
		dim      = 100
	)

	mean1 := randVec(dim, 1)
	mean2 := randVec(dim, 1)
	x1 := randData(numSets1, setLen1, mean1, 1)
	x2 := randData(numSets2, setLen2, mean2, 1)
	cpos := 1 / float64(numSets1)
	cneg := 1 / float64(numSets2)
	x, y, cost := merge(x1, x2, cpos, cneg)

	for i, j := range rand.Perm(len(x)) {
		x[i], x[j] = x[j], x[i]
		y[i], y[j] = y[j], y[i]
	}

	w, err := setsvm.Train(x, y, cost)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(w)
}

func randVec(n int, sigma float64) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = sigma * rand.NormFloat64()
	}
	return x
}

func randData(numSets, setLen int, mean []float64, sigma float64) [][][]float64 {
	x := make([][][]float64, numSets)
	for i := range x {
		x[i] = make([][]float64, setLen)
		for j := range x[i] {
			x[i][j] = make([]float64, len(mean))
			for k := range x[i][j] {
				x[i][j][k] = mean[k] + sigma*rand.NormFloat64()
			}
		}
	}
	return x
}

func merge(pos, neg [][][]float64, cpos, cneg float64) ([][][]float64, []float64, []float64) {
	x := make([][][]float64, 0, len(pos)+len(neg))
	y := make([]float64, 0, len(x))
	cost := make([]float64, 0, len(x))
	for _, xi := range pos {
		x = append(x, xi)
		y = append(y, 1)
		cost = append(cost, cpos)
	}
	for _, xi := range neg {
		x = append(x, xi)
		y = append(y, -1)
		cost = append(cost, cneg)
	}
	return x, y, cost
}
