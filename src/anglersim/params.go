package main

import (
	"container/list"

	"gonum.org/v1/gonum/mat"
)

type SimParams struct {
	TimeStepScale float64
}

// population parameters to setup the simulation - note in particular
// that the age groups are split into yearly bins by default
type PopParams struct {
	SpeciesNames *list.List
	BirthRates   *mat.Dense
	DeathRates   *mat.Dense
	numSpecies   int
	numAgeGroups int
}

func NewPopParams(
	SpeciesNames *list.List,
	BirthRates, DeathRates *mat.Dense,
) *PopParams {
	numSpecies, numAgeGroups := BirthRates.Dims()
	p := &PopParams{
		SpeciesNames: SpeciesNames,
		BirthRates:   BirthRates,
		DeathRates:   DeathRates,
		numSpecies:   numSpecies,
		numAgeGroups: numAgeGroups,
	}
	return p
}
