package main

import (
	"container/list"
	"io/ioutil"
	"log"

	"gonum.org/v1/gonum/mat"
	"google.golang.org/protobuf/proto"
)

func main() {
	const runTime float64 = 10.0
	const timeScale float64 = 0.001
	const numReals int = 100
	const seed uint64 = 42
	speciesNames := list.New()
	speciesNames.PushBack("Bream")
	speciesNames.PushBack("Cod")
	speciesNames.PushBack("Pike")
	speciesNames.PushBack("Sturgeon")
	initCounts := mat.NewVecDense(4, []float64{100000.0, 35000.0, 20000.0, 16700.0})
	densDepPowers := mat.NewVecDense(4, []float64{0.02, 0.02, 0.02, 0.02})
	birthRates := mat.NewVecDense(4, []float64{540.0, 350.0, 300.0, 700.0})
	deathRates := mat.NewVecDense(4, []float64{140.0, 250.0, 130.0, 600.0})
	predationRates := mat.NewVecDense(4, []float64{10.0, 20.0, 30.0, 6.0})
	predatorBirthIncRates := mat.NewVecDense(4, []float64{10.0, 5.0, 30.0, 30.0})
	fishingRates := mat.NewVecDense(4, []float64{40.0, 40.0, 40.0, 40.0})
	predatorMatrix := mat.NewDense(4, 4, nil)
	preyMatrix := mat.NewDense(4, 4, nil)
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			predatorMatrix.Set(i, j, 1.0)
			preyMatrix.Set(i, j, 1.0)
		}
	}
	params := NewPopParams(
		speciesNames,
		densDepPowers,
		birthRates,
		deathRates,
		predationRates,
		predatorBirthIncRates,
		fishingRates,
		predatorMatrix,
		preyMatrix,
	)
	fishPop := NewFishPop(params, initCounts)
	simParams := &SimParams{
		TimeStepScale:   timeScale,
		TotalRunTime:    runTime,
		NumRealisations: numReals,
	}
	outputCountsMessage := RunSim(simParams, fishPop, seed)
	out, err := proto.Marshal(outputCountsMessage)
	if err != nil {
		log.Fatalln("Failed to encode counts message:", err)
	}
	if err := ioutil.WriteFile("output/anglersim.data", out, 0644); err != nil {
		log.Fatalln("Failed to write counts message:", err)
	}
}
