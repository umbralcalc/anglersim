package main

import (
	"gonum.org/v1/gonum/mat"
)

type FishPop struct {
	Counts           *mat.VecDense
	Time             float64
	Params           *PopParams
	latestDeaths     *mat.VecDense
	latestBirths     *mat.VecDense
	latestPredations *mat.VecDense
}

func NewFishPop(p *PopParams) *FishPop {
	f := &FishPop{
		Counts:           mat.NewVecDense(p.numSpecies, nil),
		Time:             0.0,
		Params:           p,
		latestDeaths:     mat.NewVecDense(p.numSpecies, nil),
		latestBirths:     mat.NewVecDense(p.numSpecies, nil),
		latestPredations: mat.NewVecDense(p.numSpecies, nil),
	}
	return f
}

func (f *FishPop) StepTime(timeStep float64) {
	f.Time += timeStep
}

func (f *FishPop) ApplyDeaths() {
	f.Counts.SubVec(f.Counts, f.latestDeaths)
}

func (f *FishPop) ApplyBirths() {
	f.Counts.AddVec(f.Counts, f.latestBirths)
}

func (f *FishPop) ApplyPredations() {
	f.Counts.SubVec(f.Counts, f.latestPredations)
}
