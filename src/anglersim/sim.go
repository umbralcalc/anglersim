package main

import (
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type settable interface {
	Set(i, j int, v float64)
}

type doNothing struct{}

func (d *doNothing) Set(i, j int, v float64) {}

type Sim struct {
	SimParams          *SimParams
	FishPop            *FishPop
	expDist            *distuv.Exponential
	uniDist            *distuv.Uniform
	src                rand.Source
	timeStep           float64
	cumEventProbsDense *mat.Dense
	numSpecies         int
	numAgeGroups       int
	eventTypeLookup    []settable
}

func NewSim(simParams *SimParams, fishPop *FishPop, seed uint64) *Sim {
	numSpecies, numAgeGroups := fishPop.Params.BirthRates.Dims()
	eventTypeLookup := make([]settable, (3*numAgeGroups)+1)
	for i := range eventTypeLookup {
		if i < numAgeGroups {
			eventTypeLookup[i] = fishPop.latestBirths
		} else if i < 2*numAgeGroups {
			eventTypeLookup[i] = fishPop.latestDeaths
		} else if i < 3*numAgeGroups {
			eventTypeLookup[i] = fishPop.latestPredations
		} else {
			eventTypeLookup[i] = &doNothing{}
		}
	}
	ones := make([]float64, numSpecies*numAgeGroups)
	for i := range ones {
		ones[i] = 1.0
	}
	cumEventProbsDense := mat.NewDense(numSpecies, 3*numAgeGroups, nil)
	// loop over rows here and precompute the normalised cumulative probabilites
	// note that these can only be precomputed up to a missing normalisation that
	// comes from the drawn timestep size
	src := rand.NewSource(seed)
	s := &Sim{
		SimParams: simParams,
		FishPop:   fishPop,
		expDist: &distuv.Exponential{
			Rate: 1.0 / simParams.TimeStepScale,
			Src:  src,
		},
		uniDist: &distuv.Uniform{
			Min: 0.0,
			Max: 1.0,
			Src: src,
		},
		src:                src,
		timeStep:           0.0,
		cumEventProbsDense: cumEventProbsDense,
		numSpecies:         numSpecies,
		numAgeGroups:       numAgeGroups,
		eventTypeLookup:    eventTypeLookup,
	}
	return s
}

func (s *Sim) genNewTimeStep() {
	// note that time units are all in years
	s.timeStep = s.expDist.Rand()
	s.FishPop.StepTime(s.timeStep)
}

func (s *Sim) genCumulativeEventProbs() *mat.Dense {
	cumEventProbs := s.cumEventProbsDense
	// implement normalisation update using new timestep here!!!
	return cumEventProbs
}

// core of the model
func (s *Sim) genEvents() {
	// only an independent event each step for each population as a whole
	s.FishPop.latestBirths.Zero()
	s.FishPop.latestDeaths.Zero()
	s.FishPop.latestPredations.Zero()
	cumEventProbs := s.genCumulativeEventProbs()
	for i := 0; i < s.numSpecies; i++ {
		j := floats.Within(
			cumEventProbs.RawRowView(i),
			s.uniDist.Rand(),
		)
		s.eventTypeLookup[j].Set(i, j, 1.0)
	}
}

func (s *Sim) Step() {
	s.genNewTimeStep()
	s.genEvents()
	s.FishPop.ApplyAgeing(s.src)
	s.FishPop.ApplyDeaths()
	s.FishPop.ApplyBirths()
	s.FishPop.ApplyPredations()
}
