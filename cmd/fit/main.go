package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"

	"github.com/umbralcalc/anglersim/pkg/data"
	"github.com/umbralcalc/anglersim/pkg/population"
	"github.com/umbralcalc/stochadex/pkg/general"
	"github.com/umbralcalc/stochadex/pkg/inference"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

func main() {
	panelFile := flag.String("panel", "dat/brown_trout_panel_with_covariates.csv",
		"panel CSV with covariates")
	siteID := flag.Int("site", 1915, "site ID to fit")
	nSamples := flag.Int("n", 5000, "number of random parameter proposals")
	seed := flag.Uint64("seed", 42, "random seed")
	flag.Parse()

	// Load site data
	d := data.LoadSiteTimeSeries(*panelFile, *siteID)
	T := len(d.Years)
	log.Printf("Site %d: %d years, %d covariates", *siteID, T, d.NumCovariates)

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

		logLik := evalLogLikelihood(d, proposal)

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
	predictions := runModel(d, bestParams)
	for i, yr := range d.Years {
		obsDens := math.Exp(d.LogDensity[i][0])
		predDens := math.Exp(predictions[i])
		fmt.Printf("%-6.0f %10.4f %10.4f %10.4f\n",
			yr, obsDens, predDens, obsDens-predDens)
	}
}

// evalLogLikelihood runs the Ricker model with given params and returns
// cumulative log-likelihood of the observed data.
func evalLogLikelihood(d *data.SiteData, params []float64) float64 {
	T := len(d.Years)

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
				InitStateValues:   d.LogDensity[0],
				StateWidth:        1,
				StateHistoryDepth: 2,
				Seed:              0,
			},
			{ // 1: covariates
				Name:              "covariates",
				Params:            simulator.Params{Map: map[string][]float64{}},
				InitStateValues:   d.Covariates[0],
				StateWidth:        d.NumCovariates,
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
						"covariates":             d.Covariates[0],
						"process_noise_sd":       {procNoiseSD},
					},
				},
				ParamsFromUpstream: map[string]simulator.UpstreamConfig{
					"covariates": {Upstream: 1},
				},
				InitStateValues:   d.LogDensity[0],
				StateWidth:        1,
				StateHistoryDepth: 2,
				Seed:              1234,
			},
			{ // 3: log-likelihood comparison
				Name: "comparison",
				Params: simulator.Params{
					Map: map[string][]float64{
						"mean":               d.LogDensity[0],
						"variance":           {obsNoiseSD * obsNoiseSD},
						"latest_data_values": d.LogDensity[0],
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
		InitTimeValue:         d.Years[0],
		TimestepsHistoryDepth: 2,
	}
	settings.Init()

	// Build iterations
	iterations := []simulator.Iteration{
		&general.FromStorageIteration{Data: d.LogDensity},
		&general.FromStorageIteration{Data: d.Covariates},
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
			Data: d.Years,
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
func runModel(d *data.SiteData, params []float64) []float64 {
	T := len(d.Years)

	growthRate := params[0]
	densityDep := params[1]
	betas := params[2:5]

	logN := d.LogDensity[0][0]
	predictions := make([]float64, T)
	predictions[0] = logN

	for t := 1; t < T; t++ {
		covs := d.Covariates[t]
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
