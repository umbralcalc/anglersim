package main

import (
	"gonum.org/v1/gonum/mat"
)

type FishPop struct {
	Counts           *mat.Dense
	Ages             *mat.Dense
	Time             float64
	Params           *PopParams
	latestDeaths     *mat.Dense
	latestBirths     *mat.Dense
	latestPredations *mat.Dense
}

func NewFishPop(p *PopParams) *FishPop {
	f := &FishPop{
		Counts:           mat.NewDense(p.numSpecies, p.numAgeGroups, nil),
		Ages:             mat.NewDense(p.numSpecies, p.numAgeGroups, nil),
		Time:             0.0,
		Params:           p,
		latestDeaths:     mat.NewDense(p.numSpecies, p.numAgeGroups, nil),
		latestBirths:     mat.NewDense(p.numSpecies, p.numAgeGroups, nil),
		latestPredations: mat.NewDense(p.numSpecies, p.numAgeGroups, nil),
	}
	return f
}

func (f *FishPop) ApplyAgeing(timeStep *mat.Dense) {
	f.Ages.Add(f.Ages, timeStep)
}

func (f *FishPop) ApplyDeaths() {
	f.Counts.Sub(f.Counts, f.latestDeaths)
}

func (f *FishPop) ApplyBirths() {
	f.Counts.Add(f.Counts, f.latestBirths)
}

func (f *FishPop) ApplyPredations() {
	f.Counts.Sub(f.Counts, f.latestPredations)
}
