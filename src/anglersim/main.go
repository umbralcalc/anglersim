package main

import (
	"container/list"
	"fmt"
	"sync"
	"time"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
)

func createPopAndAgeIt(speciesNames *list.List) {
	numSpecies := speciesNames.Len()
	numAgeGroups := 10
	data := make([]float64, numSpecies*numAgeGroups)
	for i := range data {
		data[i] = 1.0
	}
	p := NewPopParams(
		speciesNames,
		mat.NewDense(numSpecies, numAgeGroups, data),
		mat.NewDense(numSpecies, numAgeGroups, data),
	)
	f := NewFishPop(p)
	// fmt.Println(f.Ages)
	src := rand.NewSource(42)
	for i := 0; i < 10000; i++ {
		f.ApplyAgeing(src)
	}
	// fmt.Println(f.Ages)
}

func main() {
	var wg sync.WaitGroup
	const numReals int = 1000
	speciesNames := list.New()
	speciesNames.PushBack("Bream")
	speciesNames.PushBack("Cod")
	speciesNames.PushBack("Pike")
	speciesNames.PushBack("Sturgeon")
	startTime := time.Now()
	for i := 0; i < numReals; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			createPopAndAgeIt(speciesNames)
		}()
	}
	wg.Wait()
	fmt.Println(time.Since(startTime))
}
