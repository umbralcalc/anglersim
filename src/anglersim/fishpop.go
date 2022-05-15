package main

import (
	"container/list"

	"gonum.org/v1/gonum/mat"
)

func getOne(i, j int, v float64) float64 {
	return 1.0
}

type FishPop struct {
	SpeciesNames *list.List
	Counts       *mat.Dense
	Ages         *mat.Dense
	Weights      *mat.Dense
	numSpecies   int
	numReals     int
	ones         *mat.Dense
}

func NewFishPop(speciesNames *list.List, numReals int) FishPop {
	numSpecies := speciesNames.Len()
	ones := mat.NewDense(numSpecies, numReals, nil)
	ones.Apply(getOne, ones)
	f := FishPop{
		SpeciesNames: speciesNames,
		Counts:       mat.NewDense(numSpecies, numReals, nil),
		Ages:         mat.NewDense(numSpecies, numReals, nil),
		Weights:      mat.NewDense(numSpecies, numReals, nil),
		numSpecies:   numSpecies,
		numReals:     numReals,
		ones:         ones,
	}
	return f
}

func (f FishPop) AgeByYear() {
	f.Ages.Add(f.Ages, f.ones)
}
