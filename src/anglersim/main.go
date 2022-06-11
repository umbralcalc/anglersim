package main

import (
	"container/list"
	"fmt"
	"sync"
	"time"
)

func createPopAndAgeIt(speciesNames *list.List) {
	numSpecies := speciesNames.Len()
	data := make([]float64, numSpecies)
	for i := range data {
		data[i] = 1.0
	}

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
