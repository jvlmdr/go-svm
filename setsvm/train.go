package setsvm

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"

	"github.com/gonum/floats"
)

const eps = 1e-9

type TerminateFunc func(epoch int, f, fPrev, g, gPrev float64, w, wPrev []float64, a, aPrev map[Index]float64) (bool, error)

// Train computes the weight vector of a linear SVM.
// The training set is made up of n sets of vectors.
// Each set x[i] has a label y[i] and contains
// an arbitrary number of vectors, but cannot be empty.
// The vectors must all have the same dimension.
// The labels y[i] are assumed to be in {-1, 1}.
func Train(x [][][]float64, y []float64, cost []float64, termfunc TerminateFunc) ([]float64, error) {
	// See http://arxiv.org/abs/1312.1743
	//
	// The problem is to find w to minimize
	//   1/2 w'w + sum_i C_i max_j max(0, 1 - y[i] dot(x[i][j], w))
	//
	// This is transformed to a constrained problem
	//   1/2 w'w + sum_i xi[i]
	// where xi[i] >= 0,
	//       xi[i] >= 1 - y[i] dot(x[i][j], w) for all j.
	//
	// The dual is to find alpha to minimize
	//   1/2 sum_{ij} sum_{kl} a[ij] a[kl] y[i] y[k] dot(x[ij], x[kl]) - sum_{ij} a[ij]
	// where a[ij] >= 0,
	//       sum_j a[ij] <= C[i].
	//
	// We take advantage of the fact that
	//   w = sum_{ij} a[ij] y[i] x[ij]
	// and therefore
	//   dot(w, y[k] x[kl]) = sum_{ij} a[ij] y[i] y[k] dot(x[ij], x[kl])
	//                      = sum_{ij} K_{ij,kl} a[ij]
	// to avoid computing and storing the kernel matrix.

	// Check that none of the example sets is empty.
	if err := errIfAnyEmpty(x); err != nil {
		return nil, err
	}
	// Get the dimension of the vectors.
	m, err := dimension(x)
	if err != nil {
		return nil, err
	}
	// Get the total number of vectors.
	all := numVectors(x)

	// Initialize dual variables and weights to zero.
	var (
		w  = make([]float64, m)
		a  = make(map[Index]float64)
		lb = math.Inf(-1)
		ub = math.Inf(1)
	)
	// Maintain sum of dual variables for each set.
	sum := make([]float64, len(x))
	// Cumulative sum of number of elements in each set.
	cdf := cumSumLens(x)

	for epoch := 0; ; epoch++ {
		log.Println("epoch", epoch)
		log.Printf("sparsity: %d / %d", len(a), cdf[len(cdf)-1])

		wPrev := make([]float64, len(w))
		copy(wPrev, w)
		aPrev := make(map[Index]float64)
		for idx, val := range a {
			aPrev[idx] = val
		}
		ubPrev := ub
		lbPrev := lb

		for iter := 0; iter < all; iter++ {
			i, j := randCumSum(cdf)
			// Consider the dual objective
			//   f(a + t e[ij]) = 1/2 h t^2 - g t + const.
			// where
			//   -g = sum_{kl} a[kl] y[i] y[k] dot(x[ij], x[kl]) - 1
			//   g = 1 - y[i] dot(x[ij], w)
			hj := floats.Dot(x[i][j], x[i][j])
			gj := 1 - y[i]*floats.Dot(x[i][j], w)
			tj := gj / hj
			if math.Abs(gj) <= eps {
				// Consider this optimal.
				continue
			}

			if gj < 0 {
				// Decrease a[ij] subject to separable constraint that a[ij] >= 0.
				if tmp := a[Index{i, j}] + tj; tmp > 0 {
					// Constraint would not be violated.
					a[Index{i, j}] = tmp
					//log.Println("decrease j, unconstr, t:", tj)
				} else {
					if a[Index{i, j}] == 0 {
						//log.Println("cannot decrease j")
						continue
					}
					// Constraint would be violated.
					// Choose tj such that a[Index{i, j}] + tj = 0.
					tj = -a[Index{i, j}]
					delete(a, Index{i, j})
					//log.Println("decrease j, constr, t:", tj)
				}
				sum[i] += tj
				floats.AddScaled(w, tj*y[i], x[i][j])
				continue
			}

			// Increase a[ij] if the constraint is not yet active.
			if sum[i] < cost[i] {
				if tmp := sum[i] + tj; tmp <= cost[i] {
					sum[i] = tmp
					//log.Println("increase j, unconstr, t:", tj)
				} else {
					// Choose tj such that sum[i] + tj = cost[i]
					tj = cost[i] - sum[i]
					sum[i] = cost[i]
					//log.Println("increase j, constr, t:", tj)
				}
				a[Index{i, j}] += tj
				floats.AddScaled(w, tj*y[i], x[i][j])
				continue
			}

			// Choose a random k != j.
			k := j
			for k != j {
				k = rand.Intn(len(x[i]))
			}
			hk := floats.Dot(x[i][k], x[i][k])
			gk := 1 - y[i]*floats.Dot(x[i][k], w)

			//	// Find some k != j such that a[ik] should also be increased.
			//	var (
			//		k      int
			//		found  bool
			//	    hk, gk float64
			//	)
			//	for _, k = range rand.Perm(len(x[i])) {
			//		if k == j {
			//			continue
			//		}
			//		hk = floats.Dot(x[i][k], x[i][k])
			//		gk = 1 - y[i]*floats.Dot(x[i][k], w)
			//		if math.Abs(gk) <= eps {
			//			continue
			//		}
			//		if gk > 0 {
			//			found = true
			//			break
			//		}
			//	}
			//	if !found {
			//		panic("could not find k != j")
			//	}

			// Solve for t in f(a + t*e[ij] - t*e[ik]).
			//   f(a) = 1/2 a' K a - 1' a
			// where K_{ij,kl} = y[i] y[k] dot(x[ij], x[kl]).
			//   f(a + t e[ij] - t e[ik])
			//   = 1/2 a' K a + a' K t (e[ij]-e[ik]) + 1/2 t^2 (e[ij]-e[ik])' K (e[ij]-e[ik]) - 1'a - t + t
			//   = 1/2 h t^2 - g t + const
			// with
			//   h = K_{ij,ij} - 2 K_{ij,ik} + K_{ik,ik}
			//     = dot(x[ij], x[ij]) - 2 dot(x[ij], x[ik]) + dot(x[ik], x[ik])
			// (y[i] does not appear above since y[i] y[i] = 1)
			//   -g = sum_{pq} a[pq] K_{pq,ij} - sum_{pq} a[pq] K_{pq,ik}
			//   g = -w' y[i] x[ij] + w' y[i] x[ik]
			hjk := hj - 2*floats.Dot(x[i][j], x[i][k]) + hk
			gjk := gj - gk
			tjk := gjk / hjk
			if tjk == 0 {
				//log.Println("no change to i or j")
				continue
			}
			if tjk < 0 {
				// Decrease a[ij], increase a[ik]. Ensure a[ij] >= 0.
				if tmp := a[Index{i, j}] + tjk; tmp > 0 {
					a[Index{i, j}] = tmp
					//log.Println("increase k, decrease j, unconstr, t:", tjk)
				} else {
					if a[Index{i, j}] == 0 {
						//log.Println("would increase k, but cannot decrease j")
						continue
					}
					// Choose tjk such that a[Index{i, j}] + tjk = 0
					tjk = -a[Index{i, j}]
					delete(a, Index{i, j})
					//log.Println("increase k, decrease j, constr, t:", tjk)
				}
				a[Index{i, j}] -= tjk
			} else {
				// Increase a[ij], decrease a[ik]. Ensure a[ik] >= 0.
				if tmp := a[Index{i, k}] - tjk; tmp > 0 {
					a[Index{i, k}] = tmp
					//log.Println("increase j, decrease k, unconstr, t:", tjk)
				} else {
					if a[Index{i, k}] == 0 {
						//log.Println("would increase j, but cannot decrease k")
						continue
					}
					// Choose tjk such that a[Index{i, k}] - tjk = 0
					tjk = a[Index{i, k}]
					delete(a, Index{i, k})
					//log.Println("increase j, decrease k, constr, t:", tjk)
				}
				a[Index{i, j}] += tjk
			}
			floats.AddScaled(w, tjk*y[i], x[i][j])
			floats.AddScaled(w, -tjk*y[i], x[i][k])
		}

		lb = dual(w, sum)
		//log.Println("dual:", lb)
		ub = primal(w, x, y, cost)
		//log.Println("primal:", ub)
		//log.Printf("bounds: [%.4g, %.4g]", lb, ub)

		term, err := termfunc(epoch+1, ub, ubPrev, lb, lbPrev, w, wPrev, a, aPrev)
		if err != nil {
			return nil, err
		}
		if term {
			return w, nil
		}
	}
}

func dual(w []float64, sum []float64) float64 {
	return -0.5*floats.Dot(w, w) + floats.Sum(sum)
}

func primal(w []float64, x [][][]float64, y []float64, c []float64) float64 {
	// Evaluate primal objective.
	f := 0.5 * floats.Dot(w, w)
	for i := range x {
		var loss float64
		for j := range x[i] {
			loss = math.Max(loss, 1-y[i]*floats.Dot(x[i][j], w))
		}
		f += c[i] * loss
	}
	return f
}

// RandCumSum returns a number i such that 0 <= i < len(s)-1.
// The likelihood of returning i is proportional to s[i+1]-s[i].
func randCumSum(s []int) (i, rem int) {
	// Example:
	// x is {3, 2, 4}, s is {0, 3, 5, 9}
	// 0, 1, 2    => 0
	// 3, 4       => 1
	// 5, 6, 7, 8 => 2
	//
	// Example:
	// x is {0, 3, 0, 0, 2, 4, 0}, s is {0, 0, 3, 3, 3, 5, 9, 9}
	// 0, 1, 2    => 1
	// 3, 4       => 4
	// 5, 6, 7, 8 => 5

	n := len(s) - 1
	// Generate a number in {0, ..., s[n]-1}.
	q := rand.Intn(s[n])
	// Find i such that s[i] <= q < s[i+1], or
	// find the minimum i such that q < s[i+1].
	i = sort.Search(n, func(i int) bool { return q < s[i+1] })
	rem = q - s[i]
	return
}

type Index struct{ Set, Elem int }

// ErrIfAnyEmpty returns an error if any of the vector sets is empty.
func errIfAnyEmpty(x [][][]float64) error {
	for i, xi := range x {
		if len(xi) == 0 {
			return fmt.Errorf("set %d is empty", i)
		}
	}
	return nil
}

// NumVectors returns the number of vectors in a collection of sets.
func numVectors(x [][][]float64) int {
	var n int
	for _, xi := range x {
		n += len(xi)
	}
	return n
}

// CumSumLens returns a list of len(x)+1 elements.
// The number of elements in x[0, ..., i-1] is s[i].
func cumSumLens(x [][][]float64) []int {
	s := make([]int, len(x)+1)
	for i, xi := range x {
		s[i+1] = s[i] + len(xi)
	}
	return s
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
