package main

import (
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type settable interface {
	SetVec(i int, v float64)
}

type doNothing struct{}

func (d *doNothing) SetVec(i int, v float64) {}

type Sim struct {
	SimParams            *SimParams
	FishPop              *FishPop
	expDist              *distuv.Exponential
	uniDist              *distuv.Uniform
	src                  rand.Source
	reciTimeStepVecDense *mat.VecDense
	timeStep             float64
	prevTimeStep         float64
	cumEventProbsDense   *mat.Dense
	numSpecies           int
	eventTypeLookup      []settable
}

func NewSim(simParams *SimParams, fishPop *FishPop, seed uint64) *Sim {
	numSpecies := fishPop.Params.BirthRates.Len()
	eventTypeLookup := []settable{
		fishPop.latestBirths,
		fishPop.latestDeaths,
		fishPop.latestPredations,
		&doNothing{},
	}
	ones := make([]float64, numSpecies)
	for i := range ones {
		ones[i] = 1.0
	}
	cumEventProbsDense := mat.NewDense(numSpecies, 4, nil)
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
		src:                  src,
		timeStep:             1.0,
		prevTimeStep:         1.0,
		reciTimeStepVecDense: mat.NewVecDense(numSpecies, ones),
		cumEventProbsDense:   cumEventProbsDense,
		numSpecies:           numSpecies,
		eventTypeLookup:      eventTypeLookup,
	}
	return s
}

func (s *Sim) genNewTimeStep() {
	// note that time units are all in years
	s.timeStep = s.expDist.Rand()
	s.FishPop.StepTime(s.timeStep)
	s.reciTimeStepVecDense.ScaleVec(s.prevTimeStep/s.timeStep, s.reciTimeStepVecDense)
}

func (s *Sim) genCumulativeEventProbs() *mat.Dense {
	// these are the main model engine computations of probabilities
	// loop over columns here and compute the normalised cumulative probabilites
	cumProbs := mat.NewVecDense(s.numSpecies, nil)
	buffer := cumProbs
	// needs implementing here
	buffer = s.FishPop.Params.BirthRates
	s.cumEventProbsDense.SetCol(0, cumProbs.RawVector().Data)
	// needs implementing here
	buffer = s.FishPop.Params.DeathRates
	cumProbs.AddVec(cumProbs, buffer)
	s.cumEventProbsDense.SetCol(1, cumProbs.RawVector().Data)
	// needs implementing here
	// predations
	cumProbs.AddVec(cumProbs, buffer)
	s.cumEventProbsDense.SetCol(2, cumProbs.RawVector().Data)
	// do nothing events
	cumProbs.AddVec(cumProbs, s.reciTimeStepVecDense)
	s.cumEventProbsDense.SetCol(3, cumProbs.RawVector().Data)
	for i := 0; i < s.numSpecies; i++ {
		// normalise each row
		buffer.DivElemVec(s.cumEventProbsDense.RowView(i), cumProbs)
		s.cumEventProbsDense.SetRow(i, buffer.RawVector().Data)
	}
	return s.cumEventProbsDense
}

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
		s.eventTypeLookup[j].SetVec(i, 1.0)
	}
}

func (s *Sim) Step() {
	s.genNewTimeStep()
	s.genEvents()
	s.FishPop.ApplyDeaths()
	s.FishPop.ApplyBirths()
	s.FishPop.ApplyPredations()
}
