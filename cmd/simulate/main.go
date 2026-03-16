package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/umbralcalc/anglersim/pkg/data"
	"github.com/umbralcalc/anglersim/pkg/simulate"
)

func main() {
	panelFile := flag.String("panel", "dat/brown_trout_panel_with_covariates.csv",
		"panel CSV with covariates")
	paramsFile := flag.String("params", "dat/hierarchical_results.csv",
		"fitted parameters CSV (batch or hierarchical)")
	outFile := flag.String("out", "dat/projections.csv", "per-site per-year projections CSV")
	summaryFile := flag.String("summary", "dat/projections_summary.csv", "per-site summary CSV")
	scenarioName := flag.String("scenario", "all",
		"scenario name: baseline, climate_1c, climate_2c, low_abstraction, drought, wq_improvement, combined_2c_oxy, or 'all'")
	horizon := flag.Int("horizon", 20, "projection horizon in years")
	nSims := flag.Int("sims", 500, "Monte Carlo trajectories per site")
	workers := flag.Int("workers", 4, "parallel workers")
	baseSeed := flag.Uint64("seed", 42, "base random seed")
	minYears := flag.Int("min-years", 10, "minimum years of data per site")
	// Custom scenario overrides
	tempShift := flag.Float64("temp-shift", 0, "custom: temperature shift (°C)")
	flowPct := flag.Float64("flow-pct", 0, "custom: flow % change (e.g., 0.15 = +15%)")
	doPct := flag.Float64("do-pct", 0, "custom: DO % change (e.g., 0.15 = +15%)")
	flag.Parse()

	// Resolve scenarios
	var scenarios []simulate.Scenario
	if *scenarioName == "all" {
		scenarios = simulate.AllScenarios()
	} else if *scenarioName == "custom" {
		scenarios = []simulate.Scenario{{
			Name:        "custom",
			Description: fmt.Sprintf("Custom: temp=%+.1f°C flow=%+.0f%% DO=%+.0f%%", *tempShift, *flowPct*100, *doPct*100),
			Modifier: simulate.CovariateModifier{
				TempShift: *tempShift,
				FlowPct:   *flowPct,
				DOPct:     *doPct,
			},
		}}
	} else {
		s, ok := simulate.ScenarioByName(*scenarioName)
		if !ok {
			names := make([]string, 0)
			for _, s := range simulate.AllScenarios() {
				names = append(names, s.Name)
			}
			log.Fatalf("Unknown scenario %q. Available: %s, all, custom", *scenarioName, strings.Join(names, ", "))
		}
		scenarios = []simulate.Scenario{s}
	}

	// Load fitted parameters
	log.Printf("Loading fitted parameters from %s ...", *paramsFile)
	siteParams := loadParams(*paramsFile)
	log.Printf("Loaded %d OK sites", len(siteParams))

	// Load panel data
	log.Printf("Loading site data from %s ...", *panelFile)
	allSites := data.LoadAllSiteTimeSeries(*panelFile)
	log.Printf("Loaded %d sites total", len(allSites))

	// Filter sites
	var siteIDs []int
	for id, sd := range allSites {
		if len(sd.Years) >= *minYears {
			if _, ok := siteParams[id]; ok {
				siteIDs = append(siteIDs, id)
			}
		}
	}
	sort.Ints(siteIDs)
	log.Printf("Projecting %d sites x %d scenarios (horizon=%d, sims=%d, workers=%d)",
		len(siteIDs), len(scenarios), *horizon, *nSims, *workers)

	// Run projections
	type job struct {
		siteID   int
		scenario simulate.Scenario
	}

	jobs := make(chan job, len(siteIDs)*len(scenarios))
	results := make(chan *simulate.SiteProjection, len(siteIDs)*len(scenarios))

	var wg sync.WaitGroup
	for w := 0; w < *workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				cfg := simulate.DefaultProjectionConfig()
				cfg.Horizon = *horizon
				cfg.NumSims = *nSims
				cfg.Seed = *baseSeed + uint64(j.siteID)
				proj := simulate.ProjectSite(allSites[j.siteID], siteParams[j.siteID], j.scenario, cfg)
				results <- proj
			}
		}()
	}

	for _, id := range siteIDs {
		for _, s := range scenarios {
			jobs <- job{siteID: id, scenario: s}
		}
	}
	close(jobs)

	go func() {
		wg.Wait()
		close(results)
	}()

	var projections []*simulate.SiteProjection
	done := 0
	total := len(siteIDs) * len(scenarios)
	for p := range results {
		done++
		if p != nil {
			projections = append(projections, p)
		}
		if done%100 == 0 || done == total {
			log.Printf("  %d/%d projections done", done, total)
		}
	}

	// Sort by scenario then site
	sort.Slice(projections, func(i, j int) bool {
		if projections[i].ScenarioName != projections[j].ScenarioName {
			return projections[i].ScenarioName < projections[j].ScenarioName
		}
		return projections[i].SiteID < projections[j].SiteID
	})

	// Write per-year projections CSV
	writeProjections(*outFile, projections)

	// Write per-site summary CSV
	writeSummary(*summaryFile, projections)

	// Print fleet-level summary
	printFleetSummary(scenarios, projections)

	log.Printf("Done. Projections: %s, Summary: %s", *outFile, *summaryFile)
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
	for _, rec := range all[1:] { // skip header
		if rec[3] != "OK" {
			continue
		}
		id, _ := strconv.Atoi(rec[0])
		p := &simulate.SiteFittedParams{
			SiteID: id,
			Mean:   make([]float64, 7),
			Std:    make([]float64, 7),
		}
		for i := range 7 {
			p.Mean[i], _ = strconv.ParseFloat(rec[4+2*i], 64)
			p.Std[i], _ = strconv.ParseFloat(rec[5+2*i], 64)
		}
		params[id] = p
	}
	return params
}

func writeProjections(path string, projections []*simulate.SiteProjection) {
	f, err := os.Create(path)
	if err != nil {
		log.Fatalf("creating projections file: %v", err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	w.Write([]string{"SITE_ID", "SCENARIO", "YEAR", "MEAN_LOG_DENSITY",
		"MEDIAN_LOG_DENSITY", "LO90", "HI90", "LO50", "HI50"})

	for _, p := range projections {
		for t := range p.ProjYears {
			w.Write([]string{
				strconv.Itoa(p.SiteID),
				p.ScenarioName,
				fmt.Sprintf("%.0f", p.ProjYears[t]),
				fmt.Sprintf("%.6f", p.MeanLogN[t]),
				fmt.Sprintf("%.6f", p.MedianLogN[t]),
				fmt.Sprintf("%.6f", p.Lo90[t]),
				fmt.Sprintf("%.6f", p.Hi90[t]),
				fmt.Sprintf("%.6f", p.Lo50[t]),
				fmt.Sprintf("%.6f", p.Hi50[t]),
			})
		}
	}
	w.Flush()
}

func writeSummary(path string, projections []*simulate.SiteProjection) {
	f, err := os.Create(path)
	if err != nil {
		log.Fatalf("creating summary file: %v", err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	w.Write([]string{"SITE_ID", "SCENARIO", "BASELINE_MEAN_LOG_DENSITY",
		"PROJECTED_MEAN_LOG_DENSITY", "DENSITY_CHANGE_PCT",
		"EXTINCTION_PROB", "RECOVERY_PROB"})

	for _, p := range projections {
		h := len(p.MeanLogN) - 1
		w.Write([]string{
			strconv.Itoa(p.SiteID),
			p.ScenarioName,
			fmt.Sprintf("%.6f", p.BaselineMeanLogN),
			fmt.Sprintf("%.6f", p.MeanLogN[h]),
			fmt.Sprintf("%.2f", p.MeanDensityChangePct),
			fmt.Sprintf("%.4f", p.ExtinctionProb),
			fmt.Sprintf("%.4f", p.RecoveryProb),
		})
	}
	w.Flush()
}

func printFleetSummary(scenarios []simulate.Scenario, projections []*simulate.SiteProjection) {
	// Group by scenario
	byScenario := make(map[string][]*simulate.SiteProjection)
	for _, p := range projections {
		byScenario[p.ScenarioName] = append(byScenario[p.ScenarioName], p)
	}

	fmt.Printf("\n%-22s %6s %10s %10s %10s %10s %10s\n",
		"Scenario", "Sites", "MedDens%", "Decline%", "Crit50%", "MeanExtP", "MeanRecP")
	fmt.Println(strings.Repeat("-", 82))

	for _, s := range scenarios {
		projs := byScenario[s.Name]
		if len(projs) == 0 {
			continue
		}

		n := len(projs)
		changes := make([]float64, n)
		declining := 0
		critical := 0
		sumExtP := 0.0
		sumRecP := 0.0

		for i, p := range projs {
			changes[i] = p.MeanDensityChangePct
			if p.MeanDensityChangePct < 0 {
				declining++
			}
			if p.MeanDensityChangePct < -50 {
				critical++
			}
			sumExtP += p.ExtinctionProb
			sumRecP += p.RecoveryProb
		}

		sort.Float64s(changes)
		medChange := changes[n/2]

		fmt.Printf("%-22s %6d %9.1f%% %9.1f%% %9.1f%% %10.4f %10.4f\n",
			s.Name, n, medChange,
			100*float64(declining)/float64(n),
			100*float64(critical)/float64(n),
			sumExtP/float64(n),
			sumRecP/float64(n))
	}
	fmt.Println()
}
