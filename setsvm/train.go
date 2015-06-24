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

type Index struct{ Set, Elem int }

type Dual []map[int]float64

func (a Dual) Len() int {
	var n int
	for i := range a {
		n += len(a[i])
	}
	return n
}

// TerminateFunc is the function type for deciding to terminate training.
// f is the current primal objective value.
// g is the current dual objective value.
type TerminateFunc func(epoch int, f, g float64, w []float64, a Dual) (bool, error)

// Train computes the weight vector of a linear SVM.
// The training set is made up of n sets of vectors.
// Each set x[i] has a label y[i] and contains
// an arbitrary number of vectors, but cannot be empty.
// The vectors must all have the same dimension.
// The labels y[i] are assumed to be in {-1, 1}.
func Train(x []Set, y []float64, cost []float64, termfunc TerminateFunc, debug bool) ([]float64, error) {
	// See http://arxiv.org/abs/1312.1743
	//
	// The problem is to find w to minimize
	//   1/2 w'w + sum_i C_i max_j max(0, 1 - y[i] dot(x[ij], w))
	//
	// This is transformed to a constrained problem
	//   1/2 w'w + sum_i xi[i]
	// where xi[i] >= 0,
	//       xi[i] >= 1 - y[i] dot(x[ij], w) for all j.
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
	m := dimension(x)
	// Get the total number of vectors.
	all := numExamples(x)

	// Initialize dual variables and weights to zero.
	var (
		w  = make([]float64, m)
		lb = math.Inf(-1)
		ub = math.Inf(1)
	)
	var a Dual = make([]map[int]float64, len(x))
	for i := range x {
		a[i] = make(map[int]float64)
	}
	// Maintain sum of dual variables for each set.
	sum := make([]float64, len(x))
	// Cumulative sum of number of elements in each set.
	cdf := cumSumLens(x)

	for epoch := 0; ; epoch++ {
		log.Println("epoch", epoch)
		log.Printf("sparsity: %d / %d", a.Len(), cdf[len(cdf)-1])

		for iter := 0; iter < all; iter++ {
			i, j := randCumSum(cdf)
			//	i := rand.Intn(len(x))
			//	j := rand.Intn(x[i].Len())
			if debug {
				log.Printf("set i = %d, example j = %d", i, j)
			}
			// Consider the dual objective
			//   f(a + t e[ij]) = 1/2 h t^2 - g t + const.
			// where
			//   -g = sum_{kl} a[kl] y[i] y[k] dot(x[ij], x[kl]) - 1
			//   g = 1 - y[i] dot(x[ij], w)
			hj := floats.Dot(x[i].At(j), x[i].At(j))
			if math.Abs(hj) < eps {
				// This can be avoided by adding a bias element to x.
				log.Println("cannot divide by zero")
				continue
			}
			gj := 1 - y[i]*floats.Dot(x[i].At(j), w)
			tj := gj / hj

			//	if math.Abs(gj) <= eps {
			if gj == 0 {
				// Consider this optimal.
				if debug {
					log.Print("optimal in j")
				}
				continue
			}

			if gj < 0 {
				// Decrease a[ij] subject to separable constraint that a[ij] >= 0.
				if tmp := a[i][j] + tj; tmp > 0 {
					// Constraint would not be violated.
					a[i][j] = tmp
					if debug {
						log.Printf("decrease j, unconstr, t: %.6g", tj)
					}
				} else {
					if a[i][j] == 0 {
						if debug {
							log.Print("cannot decrease j")
						}
						continue
					}
					// Constraint would be violated.
					// Choose tj such that a[i][j] + tj = 0.
					tj = -a[i][j]
					delete(a[i], j)
					if debug {
						log.Printf("decrease j, constr, t: %.6g", tj)
					}
				}
				sum[i] += tj
				if debug {
					log.Printf("a[ij] = %.6g, sum[i] = %.6g", a[i][j], sum[i])
				}
				floats.AddScaled(w, tj*y[i], x[i].At(j))
				continue
			}

			// Increase a[ij] if the constraint is not yet active.
			if sum[i] < cost[i] {
				if tmp := sum[i] + tj; tmp <= cost[i] {
					sum[i] = tmp
					if debug {
						log.Printf("increase j, unconstr, t: %.6g", tj)
					}
				} else {
					// Choose tj such that sum[i] + tj = cost[i]
					tj = cost[i] - sum[i]
					sum[i] = cost[i]
					if debug {
						log.Printf("increase j, constr, t: %.6g", tj)
					}
				}
				a[i][j] += tj
				if debug {
					log.Printf("a[ij] = %.6g, sum[i] = %.6g", a[i][j], sum[i])
				}
				floats.AddScaled(w, tj*y[i], x[i].At(j))
				continue
			}

			// Sum constraint is active and there is only one element in the sum.
			if x[i].Len() <= 1 {
				if debug {
					log.Print("cannot increase j")
				}
				continue
			}

			// Would increase a[ij] if the sum constraint was not active.
			// Jointly modify a[ij] and a[ik] for some k.
			var k int
			if a[i][j] == 0 {
				// If a[ij] is zero, then must find a[ik] that is non-zero to consider.
				// If both a[ij] and a[ik] were zero, could not decrease either.
				if len(a[i]) == 0 {
					// There are no non-zero elements.
					if debug {
						log.Print("could not find k != j that is non-zero")
					}
					continue
				}
				// Assume that a[i][j] == 0 implies that j is not in a[i].
				k = randKey(a[i])
			} else {
				// Choose a random k != j.
				k = j
				for k != j {
					k = rand.Intn(x[i].Len())
				}
			}
			if debug {
				log.Print("example pair j, k = %d, %d", j, k)
			}

			hk := floats.Dot(x[i].At(k), x[i].At(k))
			gk := 1 - y[i]*floats.Dot(x[i].At(k), w)
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
			hjk := hj - 2*floats.Dot(x[i].At(j), x[i].At(k)) + hk
			if math.Abs(hjk) < eps {
				log.Println("cannot divide by zero")
				continue
			}
			gjk := gj - gk
			tjk := gjk / hjk

			if tjk == 0 {
				if debug {
					log.Print("optimal in j and k")
				}
				continue
			}

			if tjk < 0 {
				// Decrease a[ij], increase a[ik]. Ensure a[ij] >= 0.
				if tmp := a[i][j] + tjk; tmp > 0 {
					a[i][j] = tmp
					if debug {
						log.Printf("increase k, decrease j, unconstr, t: %.6g", tjk)
					}
				} else {
					if a[i][j] == 0 {
						if debug {
							log.Print("would increase k, but cannot decrease j")
						}
						continue
					}
					// Choose tjk such that a[i][j] + tjk = 0
					tjk = -a[i][j]
					delete(a[i], j)
					if debug {
						log.Printf("increase k, decrease j, constr, t: %.6g", tjk)
					}
				}
				a[i][k] -= tjk
			} else {
				// Increase a[ij], decrease a[ik]. Ensure a[ik] >= 0.
				if tmp := a[i][k] - tjk; tmp > 0 {
					a[i][k] = tmp
					if debug {
						log.Printf("increase j, decrease k, unconstr, t:, %.6g", tjk)
					}
				} else {
					if a[i][k] == 0 {
						if debug {
							log.Print("would increase j, but cannot decrease k")
						}
						continue
					}
					// Choose tjk such that a[i][k] - tjk = 0
					tjk = a[i][k]
					delete(a[i], k)
					if debug {
						log.Printf("increase j, decrease k, constr, t: %.6g", tjk)
					}
				}
				a[i][j] += tjk
			}
			if debug {
				log.Printf("a[ij] = %.6g, a[ik] = %.6g", a[i][j], a[i][k])
			}
			floats.AddScaled(w, tjk*y[i], x[i].At(j))
			floats.AddScaled(w, -tjk*y[i], x[i].At(k))
		}

		lb = dual(w, sum)
		//log.Println("dual:", lb)
		ub = primal(w, x, y, cost)
		//log.Println("primal:", ub)
		//log.Printf("bounds: [%.4g, %.4g]", lb, ub)

		term, err := termfunc(epoch+1, ub, lb, w, a)
		if err != nil {
			return nil, err
		}
		if term {
			return w, nil
		}
	}
}

func randKey(a map[int]float64) int {
	for i := range a {
		return i
	}
	panic("empty map")
}

func dual(w []float64, sum []float64) float64 {
	return -0.5*floats.Dot(w, w) + floats.Sum(sum)
}

func primal(w []float64, x []Set, y []float64, c []float64) float64 {
	// Evaluate primal objective.
	f := 0.5 * floats.Dot(w, w)
	for i := range x {
		var loss float64
		for j := 0; j < x[i].Len(); j++ {
			loss = math.Max(loss, 1-y[i]*floats.Dot(x[i].At(j), w))
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

// ErrIfAnyEmpty returns an error if any of the vector sets is empty.
func errIfAnyEmpty(x []Set) error {
	for i := range x {
		if x[i].Len() == 0 {
			return fmt.Errorf("set %d is empty", i)
		}
	}
	return nil
}

// Dimension returns the dimension of
// the vectors in a set of example sets.
// Returns an error if vectors of different dimension are found
// or there are no vectors.
func dimension(x []Set) int {
	if len(x) == 0 {
		panic("empty")
	}
	var n int
	for i, xi := range x {
		ni := xi.Dim()
		if i == 0 {
			n = ni
			continue
		}
		if ni != n {
			panic(fmt.Sprintf("vector dims: found %d and %d", n, ni))
		}
	}
	return n
}

func numExamples(x []Set) int {
	var n int
	for _, xi := range x {
		n += xi.Len()
	}
	return n
}

// CumSumLens returns a list of len(x) elements.
// The total number of elements in x[0..i] is s[i].
func cumSumLens(x []Set) []int {
	s := make([]int, len(x)+1)
	for i := range x {
		s[i+1] = s[i] + x[i].Len()
	}
	return s
}
