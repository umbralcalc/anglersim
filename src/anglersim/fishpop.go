package main

import (
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type FishPop struct {
	Counts              *mat.Dense
	Time                float64
	Params              *PopParams
	latestDeaths        *mat.Dense
	latestBirths        *mat.Dense
	latestPredations    *mat.Dense
	ageBinWidthYears    float64
	ageingTimescale     float64
	timeUntilNextAgeing float64
}

func NewFishPop(p *PopParams) *FishPop {
	// choose a sensible number for the ageing timescale based on the
	// shortest birth timescale found in all of the populations
	sensibleAgeingTimescale := 1.0 / mat.Max(p.BirthRates)
	f := &FishPop{
		Counts:              mat.NewDense(p.numSpecies, p.numAgeGroups, nil),
		Time:                0.0,
		Params:              p,
		latestDeaths:        mat.NewDense(p.numSpecies, p.numAgeGroups, nil),
		latestBirths:        mat.NewDense(p.numSpecies, p.numAgeGroups, nil),
		latestPredations:    mat.NewDense(p.numSpecies, p.numAgeGroups, nil),
		ageBinWidthYears:    1.0,
		ageingTimescale:     sensibleAgeingTimescale,
		timeUntilNextAgeing: sensibleAgeingTimescale,
	}
	return f
}

func (f *FishPop) StepTime(timeStep float64) {
	f.Time += timeStep
	f.timeUntilNextAgeing -= timeStep
}

func (f *FishPop) ApplyAgeing(src rand.Source) {
	if f.timeUntilNextAgeing < 0.0 {
		return
	}
	agedCounts := f.Counts
	distBinom := distuv.Binomial{
		N:   1.0,
		P:   (f.ageingTimescale - f.timeUntilNextAgeing) / f.ageBinWidthYears,
		Src: src,
	}
	// sigh... this line is probably so slow....
	agedCounts.Apply(
		func(i int, j int, c float64) float64 {
			distBinom.N = c
			return distBinom.Rand()
		},
		f.Counts,
	)
	f.Counts.Sub(f.Counts, agedCounts)
	agedCounts.Augment(
		mat.NewDense(f.Params.numSpecies, 1, nil),
		agedCounts.Slice(0, f.Params.numSpecies, 0, f.Params.numAgeGroups-1),
	)
	f.Counts.Add(f.Counts, agedCounts)
	f.timeUntilNextAgeing = f.ageingTimescale
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
