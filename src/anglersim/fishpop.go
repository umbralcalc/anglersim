package main

import (
	"gonum.org/v1/gonum/mat"
)

type FishPop struct {
	Counts          *mat.VecDense
	Time            float64
	Params          *PopParams
	latestIncreases *mat.VecDense
	latestDecreases *mat.VecDense
}

func NewFishPop(p *PopParams, initCounts *mat.VecDense) *FishPop {
	f := &FishPop{
		Counts:          initCounts,
		Time:            0.0,
		Params:          p,
		latestIncreases: mat.NewVecDense(p.numSpecies, nil),
		latestDecreases: mat.NewVecDense(p.numSpecies, nil),
	}
	return f
}

func (f *FishPop) StepTime(timeStep float64) {
	f.Time += timeStep
}

func (f *FishPop) ApplyUpdates() {
	f.Counts.SubVec(f.Counts, f.latestDecreases)
	f.Counts.AddVec(f.Counts, f.latestIncreases)
}
