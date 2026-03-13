package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"

	"github.com/umbralcalc/anglersim/pkg/data"
	"github.com/umbralcalc/anglersim/pkg/inference"
)

func main() {
	panelFile := flag.String("panel", "dat/brown_trout_panel_with_covariates.csv",
		"panel CSV with covariates")
	siteID := flag.Int("site", 1915, "site ID to fit")
	nParticles := flag.Int("particles", 200, "number of SMC particles")
	nRounds := flag.Int("rounds", 3, "number of iterative importance sampling rounds")
	seed := flag.Uint64("seed", 42, "random seed")
	posteriorCSV := flag.String("posterior-csv", "", "output file for posterior particle cloud")
	predictionsCSV := flag.String("predictions-csv", "", "output file for predictions with uncertainty")
	verbose := flag.Bool("verbose", false, "print per-round diagnostics")
	flag.Parse()

	// Load site data
	d := data.LoadSiteTimeSeries(*panelFile, *siteID)
	T := len(d.Years)
	log.Printf("Site %d: %d years, %d covariates", *siteID, T, d.NumCovariates)

	// Configure and run SMC
	config := inference.DefaultSMCConfig()
	config.NumParticles = *nParticles
	config.NumRounds = *nRounds
	config.Seed = *seed
	config.Verbose = *verbose

	result := inference.RunSMC(d, config)

	// Print posterior summary
	fmt.Printf("\n=== SMC Posterior Summary (N=%d particles, %d rounds, T=%d years) ===\n",
		*nParticles, *nRounds, T)
	fmt.Printf("Log marginal likelihood: %.4f\n\n", result.LogMarginalLik)
	fmt.Printf("%-22s %10s %10s %10s %10s %10s\n",
		"Parameter", "Mean", "Std", "2.5%", "Median", "97.5%")
	fmt.Println("--------------------------------------------------------------------------")

	for i, name := range result.ParamNames {
		// Extract this param's values across particles
		paramVals := make([]float64, len(result.ParticleParams))
		for p := range result.ParticleParams {
			paramVals[p] = result.ParticleParams[p][i]
		}
		q := inference.WeightedQuantiles(paramVals, result.Weights,
			[]float64{0.025, 0.5, 0.975})
		fmt.Printf("%-22s %10.4f %10.4f %10.4f %10.4f %10.4f\n",
			name, result.PosteriorMean[i], result.PosteriorStd[i],
			q[0], q[1], q[2])
	}

	// Predictions with uncertainty
	fmt.Printf("\n%-6s %10s %10s %10s %10s\n",
		"YEAR", "OBS_DENS", "PRED_MED", "PRED_2.5%", "PRED_97.5%")
	fmt.Println("-------------------------------------------------------")
	for t := range T {
		obsDens := math.Exp(d.LogDensity[t][0])
		q := inference.WeightedQuantiles(result.Predictions[t], result.Weights,
			[]float64{0.025, 0.5, 0.975})
		fmt.Printf("%-6.0f %10.4f %10.4f %10.4f %10.4f\n",
			d.Years[t], obsDens, math.Exp(q[1]), math.Exp(q[0]), math.Exp(q[2]))
	}

	// Optional CSV outputs
	if *posteriorCSV != "" {
		writePosteriorCSV(*posteriorCSV, result)
		log.Printf("Posterior written to %s", *posteriorCSV)
	}
	if *predictionsCSV != "" {
		writePredictionsCSV(*predictionsCSV, d, result)
		log.Printf("Predictions written to %s", *predictionsCSV)
	}
}

func writePosteriorCSV(path string, result *inference.SMCResult) {
	f, err := os.Create(path)
	if err != nil {
		log.Fatalf("creating posterior CSV: %v", err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	header := append(result.ParamNames, "weight")
	w.Write(header)

	for p := range result.ParticleParams {
		row := make([]string, len(result.ParamNames)+1)
		for i, v := range result.ParticleParams[p] {
			row[i] = strconv.FormatFloat(v, 'f', 6, 64)
		}
		row[len(result.ParamNames)] = strconv.FormatFloat(result.Weights[p], 'e', 6, 64)
		w.Write(row)
	}
}

func writePredictionsCSV(path string, d *data.SiteData, result *inference.SMCResult) {
	f, err := os.Create(path)
	if err != nil {
		log.Fatalf("creating predictions CSV: %v", err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	w.Write([]string{"YEAR", "OBS_DENSITY", "PRED_MEDIAN", "PRED_2.5", "PRED_97.5"})

	for t := range len(d.Years) {
		obsDens := math.Exp(d.LogDensity[t][0])
		q := inference.WeightedQuantiles(result.Predictions[t], result.Weights,
			[]float64{0.025, 0.5, 0.975})
		w.Write([]string{
			strconv.FormatFloat(d.Years[t], 'f', 0, 64),
			strconv.FormatFloat(obsDens, 'f', 6, 64),
			strconv.FormatFloat(math.Exp(q[1]), 'f', 6, 64),
			strconv.FormatFloat(math.Exp(q[0]), 'f', 6, 64),
			strconv.FormatFloat(math.Exp(q[2]), 'f', 6, 64),
		})
	}
}
