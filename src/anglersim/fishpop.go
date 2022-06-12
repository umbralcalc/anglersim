package main

import (
	"gonum.org/v1/gonum/mat"
)

type FishPop struct {
	Counts          *mat.Dense
	Time            *mat.VecDense
	Params          *PopParams
	latestIncreases *mat.VecDense
	latestDecreases *mat.VecDense
}

func NewFishPop(p *PopParams, initCounts *mat.Dense) *FishPop {
	_, numThreads := initCounts.Dims()
	f := &FishPop{
		Counts:          initCounts,
		Time:            mat.NewVecDense(numThreads, nil),
		Params:          p,
		latestIncreases: mat.NewVecDense(p.numSpecies, nil),
		latestDecreases: mat.NewVecDense(p.numSpecies, nil),
	}
	return f
}

func (f *FishPop) StepTime(timeStep float64, threadNum int) {
	f.Time.SetVec(threadNum, f.Time.AtVec(threadNum)+timeStep)
}

func (f *FishPop) ApplyUpdates(threadNum int) {
	f.latestDecreases.SubVec(f.Counts.ColView(threadNum), f.latestDecreases)
	f.latestIncreases.AddVec(f.latestDecreases, f.latestIncreases)
	f.Counts.SetCol(threadNum, f.latestIncreases.RawVector().Data)
}
