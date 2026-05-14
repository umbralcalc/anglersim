//go:build js && wasm

// register_step is the fish climate-vulnerability widget compiled as a
// WebAssembly module. It registers `stepSimulation` on the JS global
// and blocks forever so the Go runtime stays alive to service per-step
// calls from dexetera's runtime/worker.js.
//
// Build with the codegen-emitted app/fish/build.sh or directly:
//
//	GOOS=js GOARCH=wasm go build -o app/fish/src/main.wasm ./app/cmd/fish/register_step
package main

import (
	"github.com/umbralcalc/anglersim/app/pkg/fish"
	"github.com/umbralcalc/dexetera/pkg/simio"
)

func main() {
	simio.RegisterStep(fish.NewConfig())
}
