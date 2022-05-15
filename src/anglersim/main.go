package main

import (
	"container/list"
	"fmt"
)

func main() {
	speciesNames := list.New()
	speciesNames.PushBack("Bream")
	speciesNames.PushBack("Cod")
	speciesNames.PushBack("Pike")
	speciesNames.PushBack("Sturgeon")
	f := NewFishPop(speciesNames, 10)
	fmt.Println(f.Ages)
	f.AgeByYear()
	fmt.Println(f.Ages)
}
