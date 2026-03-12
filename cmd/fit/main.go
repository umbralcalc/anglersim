package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"strconv"

	"github.com/umbralcalc/anglersim/pkg/population"
	"github.com/umbralcalc/stochadex/pkg/general"
	"github.com/umbralcalc/stochadex/pkg/inference"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

// siteData holds the time series for a single site.
type siteData struct {
	years       []float64
	logDensity  [][]float64 // [T][1] — observed log-density per year
	covariates  [][]float64 // [T][K] — environmental covariates per year
	numCovariates int
}

func main() {
	panelFile := flag.String("panel", "dat/brown_trout_panel_with_covariates.csv",
		"panel CSV with covariates")
	siteID := flag.Int("site", 1915, "site ID to fit")
	nSamples := flag.Int("n", 5000, "number of random parameter proposals")
	seed := flag.Uint64("seed", 42, "random seed")
	flag.Parse()

	// Load site data
	data := loadSiteData(*panelFile, *siteID)
	T := len(data.years)
	log.Printf("Site %d: %d years, %d covariates", *siteID, T, data.numCovariates)

	// Parameter space: [growth_rate, density_dependence, beta_flow, beta_temp, beta_do, process_noise_sd, obs_noise_sd]
	paramNames := []string{"growth_rate", "density_dependence",
		"beta_flow", "beta_temp", "beta_do",
		"process_noise_sd", "obs_noise_sd"}
	nParams := len(paramNames)

	// Prior ranges for random search
	type paramRange struct{ lo, hi float64 }
	ranges := []paramRange{
		{-1.0, 3.0},   // growth_rate
		{0.0, 50.0},   // density_dependence (in log-density space, so can be large)
		{-0.5, 0.5},   // beta_flow
		{-0.5, 0.5},   // beta_temp
		{-0.5, 0.5},   // beta_do
		{0.01, 1.0},   // process_noise_sd
		{0.01, 1.0},   // obs_noise_sd
	}

	rng := rand.New(rand.NewPCG(*seed, *seed+1))

	bestLogLik := math.Inf(-1)
	bestParams := make([]float64, nParams)

	for i := range *nSamples {
		// Sample parameters uniformly from prior ranges
		proposal := make([]float64, nParams)
		for j := range nParams {
			proposal[j] = ranges[j].lo + rng.Float64()*(ranges[j].hi-ranges[j].lo)
		}

		logLik := evalLogLikelihood(data, proposal)

		if logLik > bestLogLik {
			bestLogLik = logLik
			copy(bestParams, proposal)
			if (i+1) <= 100 || (i+1)%500 == 0 {
				log.Printf("  [%d] new best loglik=%.4f params=%v",
					i+1, bestLogLik, fmtParams(paramNames, bestParams))
			}
		}
	}

	fmt.Println()
	fmt.Println("=== Best fit (maximum likelihood) ===")
	fmt.Printf("Log-likelihood: %.4f\n", bestLogLik)
	for i, name := range paramNames {
		fmt.Printf("  %-22s = %.6f\n", name, bestParams[i])
	}

	// Print model predictions vs observed
	fmt.Println()
	fmt.Printf("%-6s %10s %10s %10s\n", "YEAR", "OBS_DENS", "PRED_DENS", "RESID")
	predictions := runModel(data, bestParams)
	for i, yr := range data.years {
		obsDens := math.Exp(data.logDensity[i][0])
		predDens := math.Exp(predictions[i])
		fmt.Printf("%-6.0f %10.4f %10.4f %10.4f\n",
			yr, obsDens, predDens, obsDens-predDens)
	}
}

// evalLogLikelihood runs the Ricker model with given params and returns
// cumulative log-likelihood of the observed data.
func evalLogLikelihood(data *siteData, params []float64) float64 {
	T := len(data.years)

	growthRate := params[0]
	densityDep := params[1]
	betas := params[2:5] // flow, temp, do
	procNoiseSD := params[5]
	obsNoiseSD := params[6]

	// Build inner simulation settings
	settings := &simulator.Settings{
		Iterations: []simulator.IterationSettings{
			{ // 0: observed data
				Name:              "observed_data",
				Params:            simulator.Params{Map: map[string][]float64{}},
				InitStateValues:   data.logDensity[0],
				StateWidth:        1,
				StateHistoryDepth: 2,
				Seed:              0,
			},
			{ // 1: covariates
				Name:              "covariates",
				Params:            simulator.Params{Map: map[string][]float64{}},
				InitStateValues:   data.covariates[0],
				StateWidth:        data.numCovariates,
				StateHistoryDepth: 2,
				Seed:              0,
			},
			{ // 2: ricker model
				Name: "population",
				Params: simulator.Params{
					Map: map[string][]float64{
						"growth_rate":            {growthRate},
						"density_dependence":     {densityDep},
						"covariate_coefficients": betas,
						"covariates":             data.covariates[0],
						"process_noise_sd":       {procNoiseSD},
					},
				},
				ParamsFromUpstream: map[string]simulator.UpstreamConfig{
					"covariates": {Upstream: 1},
				},
				InitStateValues:   data.logDensity[0],
				StateWidth:        1,
				StateHistoryDepth: 2,
				Seed:              1234,
			},
			{ // 3: log-likelihood comparison
				Name: "comparison",
				Params: simulator.Params{
					Map: map[string][]float64{
						"mean":               data.logDensity[0],
						"variance":           {obsNoiseSD * obsNoiseSD},
						"latest_data_values": data.logDensity[0],
						"cumulative":         {1},
						"burn_in_steps":      {0},
					},
				},
				ParamsFromUpstream: map[string]simulator.UpstreamConfig{
					"mean":               {Upstream: 2},
					"latest_data_values": {Upstream: 0},
				},
				InitStateValues:   []float64{0.0},
				StateWidth:        1,
				StateHistoryDepth: 2,
				Seed:              0,
			},
		},
		InitTimeValue:         data.years[0],
		TimestepsHistoryDepth: 2,
	}
	settings.Init()

	// Build iterations
	iterations := []simulator.Iteration{
		&general.FromStorageIteration{Data: data.logDensity},
		&general.FromStorageIteration{Data: data.covariates},
		&population.RickerIteration{},
		&inference.DataComparisonIteration{
			Likelihood: &inference.NormalLikelihoodDistribution{},
		},
	}
	for i, iter := range iterations {
		iter.Configure(i, settings)
	}

	store := simulator.NewStateTimeStorage()
	implementations := &simulator.Implementations{
		Iterations:      iterations,
		OutputCondition: &simulator.EveryStepOutputCondition{},
		OutputFunction:  &simulator.StateTimeStorageOutputFunction{Store: store},
		TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
			MaxNumberOfSteps: T - 1,
		},
		TimestepFunction: &general.FromStorageTimestepFunction{
			Data: data.years,
		},
	}

	// Run simulation
	func() {
		defer func() { recover() }() // catch panics from NaN/divergence
		coordinator := simulator.NewPartitionCoordinator(settings, implementations)
		coordinator.Run()
	}()

	// Extract cumulative log-likelihood
	compStates := store.GetValues("comparison")
	if len(compStates) == 0 {
		return math.Inf(-1)
	}
	logLik := compStates[len(compStates)-1][0]
	if math.IsNaN(logLik) || math.IsInf(logLik, 0) {
		return math.Inf(-1)
	}
	return logLik
}

// runModel runs the Ricker model deterministically (zero process noise)
// and returns predicted log-densities at each year.
func runModel(data *siteData, params []float64) []float64 {
	T := len(data.years)

	growthRate := params[0]
	densityDep := params[1]
	betas := params[2:5]

	logN := data.logDensity[0][0]
	predictions := make([]float64, T)
	predictions[0] = logN

	for t := 1; t < T; t++ {
		covs := data.covariates[t]
		envEffect := 0.0
		k := len(betas)
		if len(covs) < k {
			k = len(covs)
		}
		for i := 0; i < k; i++ {
			envEffect += betas[i] * covs[i]
		}
		density := math.Exp(logN)
		logN = logN + growthRate + envEffect - densityDep*density
		predictions[t] = logN
	}
	return predictions
}

func loadSiteData(panelFile string, siteID int) *siteData {
	f, err := os.Open(panelFile)
	if err != nil {
		log.Fatalf("opening panel: %v", err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	headers, err := r.Read()
	if err != nil {
		log.Fatalf("reading header: %v", err)
	}

	idx := make(map[string]int)
	for i, h := range headers {
		idx[h] = i
	}

	var years []float64
	var logDensities [][]float64
	var covariates [][]float64

	for {
		record, err := r.Read()
		if err != nil {
			break
		}
		id, _ := strconv.Atoi(record[idx["SITE_ID"]])
		if id != siteID {
			continue
		}

		year, _ := strconv.ParseFloat(record[idx["YEAR"]], 64)
		density, _ := strconv.ParseFloat(record[idx["DENSITY"]], 64)

		// Skip rows with zero density (can't take log)
		if density <= 0 {
			continue
		}

		// Parse covariates (flow, temp, DO) — use 0 for missing
		flow := parseFloat(record[idx["MEAN_FLOW"]])
		temp := parseFloat(record[idx["MEAN_TEMP"]])
		do := parseFloat(record[idx["MEAN_DO"]])

		years = append(years, year)
		logDensities = append(logDensities, []float64{math.Log(density)})
		covariates = append(covariates, []float64{flow, temp, do})
	}

	if len(years) == 0 {
		log.Fatalf("no data found for site %d", siteID)
	}

	return &siteData{
		years:         years,
		logDensity:    logDensities,
		covariates:    covariates,
		numCovariates: 3,
	}
}

func parseFloat(s string) float64 {
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0.0
	}
	return v
}

func fmtParams(names []string, vals []float64) string {
	s := ""
	for i, name := range names {
		if i > 0 {
			s += ", "
		}
		s += fmt.Sprintf("%s=%.4f", name, vals[i])
	}
	return s
}
