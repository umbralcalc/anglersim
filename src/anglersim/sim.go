package main

import (
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type Sim struct {
	simParams *SimParams
	fishPop   *FishPop
}

func (s Sim) genNewTimeStep(src rand.Source) *mat.Dense {
	e := distuv.Exponential{
		Rate: 1.0 / s.simParams.TimeStepScale,
		Src:  src,
	}
	data := make(
		[]float64,
		s.fishPop.Params.numSpecies*s.fishPop.Params.numSubGroups,
	)
	for i := range data {
		data[i] = e.Rand()
	}
	return mat.NewDense(
		s.fishPop.Params.numSpecies,
		s.fishPop.Params.numSubGroups,
		data,
	)
}