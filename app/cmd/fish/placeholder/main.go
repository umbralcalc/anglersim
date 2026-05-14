// placeholder emits a synthetic projection_grid.json that satisfies
// the fish.Grid schema without requiring access to the NFPD data or
// fitted parameters. It exists so the wasm build can compile from a
// fresh checkout; run cmd/fish/precompute to replace the placeholder
// with the real grid before publishing.
//
//	cd app && go run ./cmd/fish/placeholder
//
// The synthetic trajectories have the qualitative shape the post is
// about — climate warming dominates, flow and DO are near-inert — so
// local widget previews convey the intended finding even before the
// real grid is computed.
package main

import (
	"encoding/json"
	"flag"
	"log"
	"math"
	"os"

	"github.com/umbralcalc/anglersim/app/pkg/fish"
)

func main() {
	outFile := flag.String("out", "./pkg/fish/projection_grid.json",
		"output JSON path")
	flag.Parse()

	climateAxis := []float64{0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0}
	flowAxis := []float64{-50, -25, 0, 25, 50}
	doAxis := []float64{-25, -12.5, 0, 12.5, 25}
	regions := []string{
		"Devon & Cornwall", "Eastern", "Kent & South London",
		"North", "North East", "Northern",
		"Wessex", "West", "Yorkshire",
	}
	bands := []string{"Stable", "Declining", "Critical >50%", "Extinct >90%"}
	maxHorizon := 30
	baseline := -3.0

	nc, nf, nd, nh := len(climateAxis), len(flowAxis), len(doAxis), maxHorizon
	nr, nb := len(regions), len(bands)

	g := &fish.Grid{
		ClimateAxis:  climateAxis,
		FlowAxis:     flowAxis,
		DOAxis:       doAxis,
		Regions:      regions,
		Bands:        bands,
		MaxHorizon:   maxHorizon,
		BaselineLogN: baseline,
		Median:       make([]float64, nc*nf*nd*nh),
		Lo90:         make([]float64, nc*nf*nd*nh),
		Hi90:         make([]float64, nc*nf*nd*nh),
		RegionalPct:  make([]float64, nc*nf*nd*nr*nh),
		BandCounts:   make([]float64, nc*nf*nd*nb*nh),
	}

	// Region-specific vulnerability multipliers — Kent & South London the
	// most exposed, Devon & Cornwall the most resilient. Loosely mirrors
	// the regional ordering in anglersim's README so widget previews
	// land the intended story before the real grid is computed.
	regionVuln := map[string]float64{
		"Kent & South London": 1.8,
		"West":                1.4,
		"Wessex":              1.3,
		"Eastern":             1.25,
		"Yorkshire":           1.0,
		"Northern":            0.9,
		"North":               0.8,
		"North East":          0.7,
		"Devon & Cornwall":    0.5,
	}

	for ci, climate := range climateAxis {
		for fi, flow := range flowAxis {
			for di, do := range doAxis {
				// Decay rate in delta-log-density units per year. Climate
				// is the dominant driver; flow and DO have ~10x smaller
				// per-unit effect, matching the post's central finding.
				perYear := -0.025 - 0.040*climate - 0.0005*(-flow) - 0.0006*(-do)
				cellBase := ((ci*nf+fi)*nd + di) * nh

				for y := 0; y < nh; y++ {
					t := float64(y + 1)
					delta := perYear * t
					g.Median[cellBase+y] = baseline + delta
					// 90% band widens linearly with horizon and warming.
					halfBand := 0.30 + 0.025*t + 0.10*climate
					g.Lo90[cellBase+y] = baseline + delta - halfBand
					g.Hi90[cellBase+y] = baseline + delta + halfBand
				}

				// Regional bars: each region's % density change at year y
				// is the cell's % change scaled by its vulnerability.
				regBase := (((ci*nf+fi)*nd + di) * nr) * nh
				for r, regionName := range regions {
					vuln := regionVuln[regionName]
					for y := 0; y < nh; y++ {
						t := float64(y + 1)
						pct := 100 * (math.Exp(perYear*t*vuln) - 1)
						g.RegionalPct[regBase+r*nh+y] = pct
					}
				}

				// Distribution bars: count of (synthetic) 790 sites in
				// each outcome band at each year.
				bandBase := (((ci*nf+fi)*nd + di) * nb) * nh
				const totalSites = 790
				for y := 0; y < nh; y++ {
					t := float64(y + 1)
					counts := [4]float64{0, 0, 0, 0}
					for _, regionName := range regions {
						vuln := regionVuln[regionName]
						pct := 100 * (math.Exp(perYear*t*vuln) - 1)
						b := bandFor(pct)
						// Approximate per-region share of 790; rounded
						// so the bars look natural at first load.
						share := totalSites / float64(len(regions))
						counts[b] += share
					}
					// Renormalise to total ~ 790.
					sum := counts[0] + counts[1] + counts[2] + counts[3]
					if sum > 0 {
						for b := range counts {
							counts[b] = counts[b] / sum * totalSites
						}
					}
					for b := 0; b < nb; b++ {
						g.BandCounts[bandBase+b*nh+y] = counts[b]
					}
				}
			}
		}
	}

	if err := g.Validate(); err != nil {
		log.Fatalf("placeholder grid failed validation: %v", err)
	}

	out, err := os.Create(*outFile)
	if err != nil {
		log.Fatalf("creating %s: %v", *outFile, err)
	}
	defer out.Close()
	enc := json.NewEncoder(out)
	if err := enc.Encode(g); err != nil {
		log.Fatalf("encoding placeholder grid: %v", err)
	}
	log.Printf("Wrote placeholder grid to %s (%d cells × %d horizon)",
		*outFile, nc*nf*nd, nh)
}

func bandFor(pctChange float64) int {
	switch {
	case pctChange <= -90:
		return 3
	case pctChange <= -50:
		return 2
	case pctChange <= -10:
		return 1
	default:
		return 0
	}
}
