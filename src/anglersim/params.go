package main

import (
	"container/list"

	"gonum.org/v1/gonum/mat"
)

type SimParams struct {
	TimeStepScale float64
}

// population parameters to setup the simulation
type PopParams struct {
	SpeciesNames *list.List
	BirthRates   *mat.VecDense
	DeathRates   *mat.VecDense
	numSpecies   int
}

func NewPopParams(
	SpeciesNames *list.List,
	BirthRates, DeathRates *mat.VecDense,
) *PopParams {
	numSpecies := BirthRates.Len()
	p := &PopParams{
		SpeciesNames: SpeciesNames,
		BirthRates:   BirthRates,
		DeathRates:   DeathRates,
		numSpecies:   numSpecies,
	}
	return p
}
