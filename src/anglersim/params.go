package main

import (
	"container/list"

	"gonum.org/v1/gonum/mat"
)

type SimParams struct {
	TimeStepScale float64
}

type PopParams struct {
	SpeciesNames *list.List
	BirthRates   *mat.Dense
	DeathRates   *mat.Dense
	numSpecies   int
	numSubGroups int
}

func NewPopParams(
	SpeciesNames *list.List,
	BirthRates, DeathRates *mat.Dense,
) PopParams {
	numSpecies, numSubGroups := BirthRates.Dims()
	p := PopParams{
		SpeciesNames: SpeciesNames,
		BirthRates:   BirthRates,
		DeathRates:   DeathRates,
		numSpecies:   numSpecies,
		numSubGroups: numSubGroups,
	}
	return p
}
