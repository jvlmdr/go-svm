package svm

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/gonum/floats"
)

const eps = 1e-9

type TerminateFunc func(epoch int, f, fPrev, g, gPrev float64, w, wPrev []float64, a, aPrev map[int]float64) (bool, error)

// Train computes the weight vector of a linear SVM.
// Each example x[i] has a label y[i].
// The vectors must all have the same dimension.
// The labels y[i] are assumed to be in {-1, 1}.
func Train(x [][]float64, y []float64, cost []float64, termfunc TerminateFunc) ([]float64, error) {
	// Get the dimension of the vectors.
	m, err := dimension(x)
	if err != nil {
		return nil, err
	}
	// Initialize dual variables and weights to zero.
	var (
		a  = make(map[int]float64)
		w  = make([]float64, m)
		lb = math.Inf(-1)
		ub = math.Inf(1)
	)

	for epoch := 0; ; epoch++ {
		log.Println("epoch", epoch)
		log.Printf("sparsity: %d / %d", len(a), len(x))

		wPrev := make([]float64, len(w))
		copy(wPrev, w)
		aPrev := make(map[int]float64)
		for i, val := range a {
			aPrev[i] = val
		}
		ubPrev := ub
		lbPrev := lb

		for iter := 0; iter < len(x); iter++ {
			i := rand.Intn(len(x))
			// Consider the dual objective
			//   f(a + t e[i]) = 1/2 h t^2 - g t + const.
			// where
			//   h = dot(x[i], x[i])
			//   -g = sum_{j} a[j] y[i] y[j] dot(x[i], x[j]) - 1
			//   g = 1 - y[i] dot(x[i], w)
			g := 1 - y[i]*floats.Dot(x[i], w)
			if math.Abs(g) <= eps {
				// Consider this optimal.
				continue
			}
			h := floats.Dot(x[i], x[i])
			t := g / h
			if t == 0 {
				// No change.
				continue
			}
			if t < 0 {
				// Decrease a[i]. Ensure that a[i] >= 0.
				if tmp := a[i] + t; tmp > 0 {
					a[i] = tmp
					//log.Println("decrease i, unconstr, t:", t)
				} else {
					if a[i] == 0 {
						// No change.
						//log.Print("cannot decrease i")
						continue
					}
					// Choose t such that a[i] + t = 0.
					t = -a[i]
					delete(a, i)
					//log.Println("decrease i, constr, t:", t)
				}
			} else {
				if tmp := a[i] + t; tmp < cost[i] {
					a[i] = tmp
					//log.Println("increase i, unconstr, t:", t)
				} else {
					if a[i] == cost[i] {
						// No change.
						//log.Print("cannot increase i")
						continue
					}
					// Choose t such that a[i] + t = cost[i].
					t = cost[i] - a[i]
					a[i] = cost[i]
					//log.Println("increase i, constr, t:", t)
				}
			}
			floats.AddScaled(w, t*y[i], x[i])
		}

		lb = dual(w, a)
		//log.Println("dual:", lb)
		ub = primal(w, x, y, cost)
		//log.Println("primal:", ub)
		//log.Printf("bounds: [%.6g, %.6g]", lb, ub)

		term, err := termfunc(epoch+1, ub, ubPrev, lb, lbPrev, w, wPrev, a, aPrev)
		if err != nil {
			return nil, err
		}
		if term {
			return w, nil
		}
	}
}

func dual(w []float64, a map[int]float64) float64 {
	// Evaluate dual objective.
	f := -0.5 * floats.Dot(w, w)
	for _, ai := range a {
		f += ai
	}
	return f
}

func primal(w []float64, x [][]float64, y []float64, c []float64) float64 {
	f := 0.5 * floats.Dot(w, w)
	for i := range x {
		f += c[i] * math.Max(0, 1-y[i]*floats.Dot(x[i], w))
	}
	return f
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
