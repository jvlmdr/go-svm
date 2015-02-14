package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/jvlmdr/go-svm/setsvm"
	"github.com/jvlmdr/go-svm/svm"
)

func main() {
	var (
		numSets1     = 1000
		numSets2     = 1000
		setLen1      = 10
		setLen2      = 1000
		dim          = 100
		sigmaMean    = float64(1)
		sigmaExample = float64(1)
		sigmaNoise   = 1e-1
		lambda       = 1e-2
	)

	u1 := randVec(dim, sigmaMean)
	u2 := randVec(dim, sigmaMean)
	x1 := randSets(numSets1, setLen1, u1, sigmaExample, sigmaNoise)
	x2 := randSets(numSets2, setLen2, u2, sigmaExample, sigmaNoise)

	x, y := mergePosNeg(x1, x2)
	cset := make([]float64, 0, len(x))
	for _ = range x1 {
		cset = append(cset, 1/lambda*1/float64(numSets1))
	}
	for _ = range x2 {
		cset = append(cset, 1/lambda*1/float64(numSets2))
	}

	cvec := make([]float64, 0, len(x))
	for _ = range x1 {
		cvec = append(cvec, 1/lambda*1/float64(numSets1*setLen1))
	}
	for _ = range x2 {
		cvec = append(cvec, 1/lambda*1/float64(numSets2*setLen2))
	}
	u, v, z := union(x, y, cvec)

	for i, j := range rand.Perm(len(x)) {
		x[i], x[j] = x[j], x[i]
		y[i], y[j] = y[j], y[i]
		cset[i], cset[j] = cset[j], cset[i]
	}

	for i, j := range rand.Perm(len(u)) {
		u[i], u[j] = u[j], u[i]
		v[i], v[j] = v[j], v[i]
		z[i], z[j] = z[j], z[i]
	}

	sets := make([]setsvm.Set, len(x))
	for i, xi := range x {
		sets[i] = setsvm.Slice(xi)
	}

	w, err := setsvm.Train(sets, y, cset,
		func(epoch int, f, fPrev, g, gPrev float64, w, wPrev []float64, a, aPrev map[setsvm.Index]float64) (bool, error) {
			log.Printf("epoch: %d, f: %.6g, g: %.6g, gap: %.6g", epoch, f, g, f-g)
			if f-g < 1e-4 {
				return true, nil
			}
			if math.Abs(f-fPrev) < 1e-6 {
				return true, nil
			}
			if math.Abs(g-gPrev) < 1e-6 {
				return true, nil
			}
			log.Printf("df: %.6g, dg: %.6g", f-fPrev, g-gPrev)
			if epoch >= 32 {
				//return false, fmt.Errorf("too many epochs: %d", epoch)
				log.Println("reach iteration limit")
				return true, nil
			}
			return false, nil
		},
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(w)

	q, err := svm.Train(svm.Slice(u), v, z,
		func(epoch int, f, fPrev, g, gPrev float64, w, wPrev []float64, a, aPrev map[int]float64) (bool, error) {
			log.Printf("epoch: %d, f: %.6g, g: %.6g, gap: %.6g", epoch, f, g, f-g)
			if f-g < 1e-4 {
				return true, nil
			}
			// Dual co-ordinate descent has linear convergence.
			// Don't bother checking if f and g are stalling.
			if epoch >= 32 {
				//return false, fmt.Errorf("too many epochs: %d", epoch)
				log.Println("reach iteration limit")
				return true, nil
			}
			return false, nil
		},
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(q)
}

func randVec(n int, sigma float64) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = sigma * rand.NormFloat64()
	}
	return x
}

func randSets(numSets, setLen int, mean []float64, sigma, noise float64) [][][]float64 {
	dim := len(mean)
	x := make([][][]float64, numSets)
	for i := range x {
		u := randVec(dim, sigma)
		x[i] = make([][]float64, setLen)
		for j := range x[i] {
			x[i][j] = make([]float64, len(mean))
			for k := range x[i][j] {
				x[i][j][k] = u[k] + noise*rand.NormFloat64()
			}
		}
	}
	return x
}

func union(x [][][]float64, y []float64, c []float64) ([][]float64, []float64, []float64) {
	var (
		u [][]float64
		v []float64
		z []float64
	)
	for i := range x {
		for j := range x[i] {
			u = append(u, x[i][j])
			v = append(v, y[i])
			z = append(z, c[i])
		}
	}
	return u, v, z
}

func mergePosNeg(pos, neg [][][]float64) ([][][]float64, []float64) {
	x := make([][][]float64, 0, len(pos)+len(neg))
	y := make([]float64, 0, len(x))
	for _, xi := range pos {
		x = append(x, xi)
		y = append(y, 1)
	}
	for _, xi := range neg {
		x = append(x, xi)
		y = append(y, -1)
	}
	return x, y
}
