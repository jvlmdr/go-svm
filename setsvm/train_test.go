package setsvm_test

import (
	"fmt"
	"log"

	"github.com/jvlmdr/go-svm/setsvm"
)

func ExampleTrain() {
	const (
		lambda = 1e-3
		z      = 10
	)
	term := func(epoch int, f, g float64, w []float64, a setsvm.Dual) (bool, error) {
		log.Printf("primal %.4g, dual %.4g", f, g)
		if epoch >= 1000 {
			return true, nil
		}
		return false, nil
	}

	var (
		x []setsvm.Set
		y []float64
		c []float64
	)
	// One positive at (1, 1).
	y = append(y, 1)
	x = append(x, setsvm.Slice([][]float64{{10, 10, z}, {1, 1, z}}))
	c = append(c, 1/lambda)
	// One negative at (-1, 0).
	y = append(y, -1)
	x = append(x, setsvm.Slice([][]float64{{-1, 0, z}, {-10, 0, z}}))
	c = append(c, 1/lambda)
	// One negative at (0, -2).
	y = append(y, -1)
	x = append(x, setsvm.Slice([][]float64{{0, -10, z}, {0, -2, z}, {-10, -10, z}}))
	c = append(c, 1/lambda)

	w, err := setsvm.Train(x, y, c, term, true)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("%.3g\n", w)
	// Output:
	// [0.8 0.4 -0.02]
}
