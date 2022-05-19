package main

import (
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

func drawUniformRandomDense(n, m int, src rand.Source) *mat.Dense {
	data := make([]float64, n*m)
	u := distuv.Uniform{
		Min: 0.0,
		Max: 1.0,
		Src: src,
	}
	for i := range data {
		data[i] = u.Rand()
	}
	return mat.NewDense(n, m, data)
}

type FishPop struct {
	Counts *mat.Dense
	Ages   *mat.Dense
	Params *PopParams
}

func NewFishPop(p *PopParams) FishPop {
	f := FishPop{
		Counts: mat.NewDense(p.numSpecies, p.numSubGroups, nil),
		Ages:   mat.NewDense(p.numSpecies, p.numSubGroups, nil),
	}
	return f
}

func (f FishPop) agePopulation(timeStep *mat.Dense) {
	f.Ages.Add(f.Ages, timeStep)
}

func (f FishPop) addBirths(timeStep *mat.Dense, src rand.Source) {
	r := drawUniformRandomDense(
		f.Params.numSpecies,
		f.Params.numSubGroups,
		src,
	)
	f.Counts.Apply(
		func(i int, j int, v float64) float64 {
			u := f.Params.BirthRates.At(i, j) * timeStep.At(i, j)
			if r.At(i, j) < (u / (1.0 + u)) {
				return v + 1.0
			} else {
				return v
			}
		},
		f.Counts,
	)
}
