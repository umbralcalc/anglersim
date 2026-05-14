// precompute scans the (climate, flow, DO) scenario grid and writes
// the aggregated projection statistics to a JSON file the fish widget
// embeds at build time. One pass over a 13×7×7 grid at 790 sites × 500
// sims × 30 years takes meaningful wall time; pick resolution flags
// accordingly.
//
//	cd app && go run ./cmd/fish/precompute \
//	    --panel ../dat/brown_trout_panel_with_covariates.csv \
//	    --params ../dat/hierarchical_results.csv \
//	    --out ./pkg/fish/projection_grid.json
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
	"sync"

	"github.com/umbralcalc/anglersim/app/pkg/fish"
	"github.com/umbralcalc/anglersim/pkg/data"
	"github.com/umbralcalc/anglersim/pkg/simulate"
)

// Outcome bands at the projection horizon, ordered to match the
// distribution_bars partition's expected output. Thresholds are in
// percent-density-change at the horizon (median trajectory).
var bandNames = []string{"Stable", "Declining", "Critical >50%", "Extinct >90%"}

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

func main() {
	panelFile := flag.String("panel", "../dat/brown_trout_panel_with_covariates.csv",
		"panel CSV with covariates")
	paramsFile := flag.String("params", "../dat/hierarchical_results.csv",
		"hierarchical fit results CSV")
	outFile := flag.String("out", "./pkg/fish/projection_grid.json",
		"output JSON grid path")
	climateMin := flag.Float64("climate-min", 0.0, "climate axis min (°C)")
	climateMax := flag.Float64("climate-max", 3.0, "climate axis max (°C)")
	climateStep := flag.Float64("climate-step", 0.5, "climate axis step (°C)")
	flowMin := flag.Float64("flow-min", -50, "flow axis min (%)")
	flowMax := flag.Float64("flow-max", 50, "flow axis max (%)")
	flowStep := flag.Float64("flow-step", 25, "flow axis step (%)")
	doMin := flag.Float64("do-min", -25, "DO axis min (%)")
	doMax := flag.Float64("do-max", 25, "DO axis max (%)")
	doStep := flag.Float64("do-step", 12.5, "DO axis step (%)")
	maxHorizon := flag.Int("max-horizon", 30, "maximum projection horizon (years)")
	nSims := flag.Int("sims", 500, "Monte Carlo trajectories per site")
	minYears := flag.Int("min-years", 10, "minimum years of data per site")
	workers := flag.Int("workers", 4, "parallel workers (per cell)")
	baseSeed := flag.Uint64("seed", 42, "base random seed")
	flag.Parse()

	climateAxis := buildAxis(*climateMin, *climateMax, *climateStep)
	flowAxis := buildAxis(*flowMin, *flowMax, *flowStep)
	doAxis := buildAxis(*doMin, *doMax, *doStep)
	log.Printf("Grid: climate=%v flow=%v DO=%v (cells=%d)",
		climateAxis, flowAxis, doAxis,
		len(climateAxis)*len(flowAxis)*len(doAxis))

	siteParams := loadParams(*paramsFile)
	log.Printf("Loaded %d OK sites from %s", len(siteParams), *paramsFile)

	allSites := data.LoadAllSiteTimeSeries(*panelFile)
	log.Printf("Loaded %d sites total from %s", len(allSites), *panelFile)

	var siteIDs []int
	for id, sd := range allSites {
		if len(sd.Years) < *minYears {
			continue
		}
		if _, ok := siteParams[id]; !ok {
			continue
		}
		siteIDs = append(siteIDs, id)
	}
	sort.Ints(siteIDs)
	log.Printf("Projecting %d sites per grid cell", len(siteIDs))

	regions := uniqueRegions(allSites, siteIDs)
	log.Printf("Regions (%d): %v", len(regions), regions)

	// Global baseline: median across sites of each site's
	// BaselineMeanLogN (mean log-density over last 5 observed years).
	siteBaselines := computeSiteBaselines(allSites, siteIDs)
	globalBaseline := medianOf(siteBaselines)
	log.Printf("Global baseline log-density (median across sites): %.4f", globalBaseline)

	nc, nf, nd, nh := len(climateAxis), len(flowAxis), len(doAxis), *maxHorizon
	nr, nb := len(regions), len(bandNames)

	grid := &fish.Grid{
		ClimateAxis:  climateAxis,
		FlowAxis:     flowAxis,
		DOAxis:       doAxis,
		Regions:      regions,
		Bands:        bandNames,
		MaxHorizon:   nh,
		BaselineLogN: globalBaseline,
		Median:       make([]float64, nc*nf*nd*nh),
		Lo90:         make([]float64, nc*nf*nd*nh),
		Hi90:         make([]float64, nc*nf*nd*nh),
		RegionalPct:  make([]float64, nc*nf*nd*nr*nh),
		BandCounts:   make([]float64, nc*nf*nd*nb*nh),
	}

	totalCells := nc * nf * nd
	cellIdx := 0
	for ci, c := range climateAxis {
		for fi, f := range flowAxis {
			for di, d := range doAxis {
				cellIdx++
				scenario := simulate.Scenario{
					Name: fmt.Sprintf("c%.2f_f%+.0f_d%+.1f", c, f, d),
					Modifier: simulate.CovariateModifier{
						TempShift: c,
						FlowPct:   f / 100.0,
						DOPct:     d / 100.0,
					},
				}
				log.Printf("[%d/%d] %s", cellIdx, totalCells, scenario.Name)

				projs := projectCell(siteIDs, allSites, siteParams, scenario,
					nh, *nSims, *workers, *baseSeed)

				aggregateCell(grid, ci, fi, di, projs, siteBaselines, regions)
			}
		}
	}

	if err := grid.Validate(); err != nil {
		log.Fatalf("grid validation failed: %v", err)
	}

	out, err := os.Create(*outFile)
	if err != nil {
		log.Fatalf("creating %s: %v", *outFile, err)
	}
	defer out.Close()
	enc := json.NewEncoder(out)
	enc.SetIndent("", "")
	if err := enc.Encode(grid); err != nil {
		log.Fatalf("encoding grid: %v", err)
	}
	log.Printf("Done. Wrote %s (%d cells × horizon %d)", *outFile, totalCells, nh)
}

// buildAxis returns evenly spaced values [min, min+step, ..., max].
// Tolerant of small floating-point drift at the upper bound.
func buildAxis(min, max, step float64) []float64 {
	if step <= 0 {
		return []float64{min}
	}
	var out []float64
	for v := min; v <= max+step*1e-6; v += step {
		out = append(out, math.Round(v*1e6)/1e6)
	}
	// Ensure max is hit exactly.
	if len(out) > 0 && math.Abs(out[len(out)-1]-max) > step*0.51 {
		out = append(out, max)
	}
	return out
}

func loadParams(path string) map[int]*simulate.SiteFittedParams {
	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("opening params: %v", err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.Comment = '#'
	all, err := r.ReadAll()
	if err != nil {
		log.Fatalf("reading params: %v", err)
	}

	params := make(map[int]*simulate.SiteFittedParams)
	for _, rec := range all[1:] {
		if rec[3] != "OK" {
			continue
		}
		id, _ := strconv.Atoi(rec[0])
		nParams := (len(rec) - 4) / 2
		p := &simulate.SiteFittedParams{
			SiteID: id,
			Mean:   make([]float64, nParams),
			Std:    make([]float64, nParams),
		}
		for i := 0; i < nParams; i++ {
			p.Mean[i], _ = strconv.ParseFloat(rec[4+2*i], 64)
			p.Std[i], _ = strconv.ParseFloat(rec[5+2*i], 64)
		}
		params[id] = p
	}
	return params
}

func uniqueRegions(allSites map[int]*data.SiteData, siteIDs []int) []string {
	seen := make(map[string]struct{})
	for _, id := range siteIDs {
		if r := allSites[id].Area; r != "" {
			seen[r] = struct{}{}
		}
	}
	out := make([]string, 0, len(seen))
	for r := range seen {
		out = append(out, r)
	}
	sort.Strings(out)
	return out
}

func computeSiteBaselines(allSites map[int]*data.SiteData, siteIDs []int) map[int]float64 {
	out := make(map[int]float64, len(siteIDs))
	for _, id := range siteIDs {
		sd := allSites[id]
		T := len(sd.Years)
		n := 5
		if T < n {
			n = T
		}
		sum := 0.0
		for i := T - n; i < T; i++ {
			sum += sd.LogDensity[i][0]
		}
		out[id] = sum / float64(n)
	}
	return out
}

func medianOf(m map[int]float64) float64 {
	if len(m) == 0 {
		return 0
	}
	vals := make([]float64, 0, len(m))
	for _, v := range m {
		vals = append(vals, v)
	}
	sort.Float64s(vals)
	return vals[len(vals)/2]
}

func projectCell(
	siteIDs []int,
	allSites map[int]*data.SiteData,
	siteParams map[int]*simulate.SiteFittedParams,
	scenario simulate.Scenario,
	horizon, nSims, workers int,
	baseSeed uint64,
) []*simulate.SiteProjection {
	jobs := make(chan int, len(siteIDs))
	results := make(chan *simulate.SiteProjection, len(siteIDs))

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for id := range jobs {
				cfg := simulate.DefaultProjectionConfig()
				cfg.Horizon = horizon
				cfg.NumSims = nSims
				cfg.Seed = baseSeed + uint64(id)
				proj := simulate.ProjectSite(allSites[id], siteParams[id], scenario, cfg)
				results <- proj
			}
		}()
	}
	for _, id := range siteIDs {
		jobs <- id
	}
	close(jobs)
	go func() { wg.Wait(); close(results) }()

	out := make([]*simulate.SiteProjection, 0, len(siteIDs))
	for p := range results {
		if p != nil {
			out = append(out, p)
		}
	}
	return out
}

// aggregateCell writes the per-cell statistics into grid for cell index
// (ci, fi, di). Per-year traces: median log-density delta-from-baseline
// re-anchored at the global baseline, plus 90% band from cross-site
// quantiles. Per-region: median % density change. Per-band: site count.
func aggregateCell(
	grid *fish.Grid,
	ci, fi, di int,
	projs []*simulate.SiteProjection,
	siteBaselines map[int]float64,
	regions []string,
) {
	nf, nd, nh := len(grid.FlowAxis), len(grid.DOAxis), grid.MaxHorizon
	nr, nb := len(regions), len(grid.Bands)
	cellBase := ((ci*nf+fi)*nd + di) * nh

	deltasPerYear := make([][]float64, nh)
	pctPerYearPerRegion := make([]map[string][]float64, nh)
	bandCountsPerYear := make([][]float64, nh)
	for y := 0; y < nh; y++ {
		deltasPerYear[y] = make([]float64, 0, len(projs))
		pctPerYearPerRegion[y] = make(map[string][]float64, nr)
		bandCountsPerYear[y] = make([]float64, nb)
	}

	for _, p := range projs {
		baseline, ok := siteBaselines[p.SiteID]
		if !ok {
			baseline = p.BaselineMeanLogN
		}
		for y := 0; y < nh && y < len(p.MedianLogN); y++ {
			delta := p.MedianLogN[y] - baseline
			deltasPerYear[y] = append(deltasPerYear[y], delta)

			// % density change in raw density space, computed from the
			// delta in log-density.
			pct := 100 * (math.Exp(delta) - 1)
			pctPerYearPerRegion[y][p.Area] = append(pctPerYearPerRegion[y][p.Area], pct)

			bandCountsPerYear[y][bandFor(pct)]++
		}
	}

	// National median + 90% band, re-anchored at the global baseline so
	// the chart's y-axis lines up with the baseline reference line.
	for y := 0; y < nh; y++ {
		ds := deltasPerYear[y]
		if len(ds) == 0 {
			continue
		}
		sort.Float64s(ds)
		grid.Median[cellBase+y] = grid.BaselineLogN + ds[len(ds)/2]
		grid.Lo90[cellBase+y] = grid.BaselineLogN + ds[int(float64(len(ds))*0.05)]
		grid.Hi90[cellBase+y] = grid.BaselineLogN + ds[int(float64(len(ds))*0.95)]
	}

	regionalBase := (((ci*nf+fi)*nd + di) * nr) * nh
	for r, regionName := range regions {
		for y := 0; y < nh; y++ {
			vals := pctPerYearPerRegion[y][regionName]
			if len(vals) == 0 {
				continue
			}
			sort.Float64s(vals)
			grid.RegionalPct[regionalBase+r*nh+y] = vals[len(vals)/2]
		}
	}

	bandBase := (((ci*nf+fi)*nd + di) * nb) * nh
	for b := 0; b < nb; b++ {
		for y := 0; y < nh; y++ {
			grid.BandCounts[bandBase+b*nh+y] = bandCountsPerYear[y][b]
		}
	}
}
