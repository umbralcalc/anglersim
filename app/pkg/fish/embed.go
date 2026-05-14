package fish

import (
	_ "embed"
	"log"
)

//go:embed projection_grid.json
var projectionGridJSON []byte

// LoadEmbeddedGrid returns the projection grid embedded into the wasm
// binary at build time. Panics if the embedded payload fails to parse —
// the grid is a build-time artefact and a corrupt embed is a hard error.
func LoadEmbeddedGrid() *Grid {
	g, err := LoadGrid(projectionGridJSON)
	if err != nil {
		log.Fatalf("fish: failed to parse embedded projection_grid.json: %v", err)
	}
	return g
}
