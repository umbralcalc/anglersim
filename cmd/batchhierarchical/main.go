package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
	"sync"

	"github.com/umbralcalc/anglersim/pkg/data"
	"github.com/umbralcalc/anglersim/pkg/inference"
)

func main() {
	stage1File := flag.String("stage1", "dat/batch_smc_results.csv",
		"Stage 1 batch results CSV")
	panelFile := flag.String("panel", "dat/brown_trout_panel_with_covariates.csv",
		"panel CSV with covariates")
	outFile := flag.String("out", "dat/hierarchical_results.csv", "output CSV path")
	workers := flag.Int("workers", 4, "number of parallel workers")
	particles := flag.Int("particles", 500, "particles per SMC run")
	rounds := flag.Int("rounds", 3, "SMC importance sampling rounds")
	baseSeed := flag.Uint64("seed", 42, "base random seed")
	minYears := flag.Int("min-years", 10, "minimum years of data per site")
	ebIter := flag.Int("iterations", 3, "empirical Bayes iterations")
	verbose := flag.Bool("verbose", false, "verbose per-site output")
	flag.Parse()

	// Stage 1: Load independent fit results
	log.Printf("Loading Stage 1 results from %s ...", *stage1File)
	stage1Sites := inference.LoadBatchResults(*stage1File)
	log.Printf("Loaded %d OK sites from Stage 1", len(stage1Sites))

	// Estimate initial hyperparameters
	hp := inference.EstimateHyperParams(stage1Sites)
	log.Printf("Initial hyperparameters:")
	logHyperParams(hp)

	// Load panel data
	log.Printf("Loading sites from %s ...", *panelFile)
	allSites := data.LoadAllSiteTimeSeries(*panelFile)
	log.Printf("Loaded %d sites total", len(allSites))

	// Filter by minimum years and collect sorted site IDs
	var siteIDs []int
	for id, sd := range allSites {
		if len(sd.Years) >= *minYears {
			siteIDs = append(siteIDs, id)
		}
	}
	sort.Ints(siteIDs)
	log.Printf("Fitting %d sites with >= %d years", len(siteIDs), *minYears)

	// Iterative empirical Bayes
	var currentSites []inference.SitePosteriorSummary

	for iter := 1; iter <= *ebIter; iter++ {
		log.Printf("=== EB iteration %d/%d (N=%d, rounds=%d, workers=%d) ===",
			iter, *ebIter, *particles, *rounds, *workers)

		priors := inference.HierarchicalPriors(hp)
		results := fitAllSites(siteIDs, allSites, priors, *particles, *rounds,
			*baseSeed, *workers, *verbose)

		// Collect summaries for hyperparameter re-estimation
		currentSites = currentSites[:0]
		paramNames := inference.DefaultSMCConfig().ParamNames
		nParams := len(paramNames)
		for _, r := range results {
			if r.res == nil {
				continue
			}
			lml := r.res.LogMarginalLik
			if math.IsNaN(lml) || math.IsInf(lml, 0) {
				continue
			}
			currentSites = append(currentSites, inference.SitePosteriorSummary{
				SiteID:     r.siteID,
				NumYears:   r.numYears,
				Mean:       r.res.PosteriorMean,
				Std:        r.res.PosteriorStd,
				LogMargLik: lml,
			})
		}

		log.Printf("  %d/%d sites OK", len(currentSites), len(siteIDs))

		// Re-estimate hyperparameters
		prevHP := *hp
		hp = inference.EstimateHyperParams(currentSites)
		log.Printf("  Updated hyperparameters:")
		logHyperParams(hp)

		// Check convergence
		muDelta := math.Abs(hp.MuBetaFlow-prevHP.MuBetaFlow) +
			math.Abs(hp.MuBetaTemp-prevHP.MuBetaTemp) +
			math.Abs(hp.MuBetaDO-prevHP.MuBetaDO)
		sigmaDelta := math.Abs(hp.SigmaBetaFlow-prevHP.SigmaBetaFlow) +
			math.Abs(hp.SigmaBetaTemp-prevHP.SigmaBetaTemp) +
			math.Abs(hp.SigmaBetaDO-prevHP.SigmaBetaDO)
		log.Printf("  Convergence: delta_mu=%.6f, delta_sigma=%.6f", muDelta, sigmaDelta)

		if muDelta < 0.001 && sigmaDelta < 0.001 && iter > 1 {
			log.Printf("  Converged after %d iterations", iter)
			break
		}

		_ = nParams
	}

	// Write final results
	writeResults(*outFile, results, hp)
	log.Printf("Done. Results in %s", *outFile)
}

type siteResult struct {
	siteID   int
	numYears int
	res      *inference.SMCResult
	err      error
}

// results is a package-level slice populated by the last fitAllSites call,
// used to write the final CSV. This avoids threading it through the EB loop.
var results []siteResult

func fitAllSites(
	siteIDs []int,
	allSites map[int]*data.SiteData,
	priors []inference.Prior,
	numParticles, numRounds int,
	baseSeed uint64,
	workers int,
	verbose bool,
) []siteResult {
	type job struct {
		siteID int
		sd     *data.SiteData
	}

	jobs := make(chan job, len(siteIDs))
	resChan := make(chan siteResult, len(siteIDs))

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				cfg := inference.SMCConfig{
					NumParticles: numParticles,
					NumRounds:    numRounds,
					Seed:         baseSeed + uint64(j.siteID),
					Priors:       priors,
					ParamNames:   inference.DefaultSMCConfig().ParamNames,
					Verbose:      verbose,
				}
				r, err := inference.RunSMCSafe(j.sd, cfg)
				resChan <- siteResult{
					siteID:   j.siteID,
					numYears: len(j.sd.Years),
					res:      r,
					err:      err,
				}
			}
		}()
	}

	for _, id := range siteIDs {
		jobs <- job{siteID: id, sd: allSites[id]}
	}
	close(jobs)

	go func() {
		wg.Wait()
		close(resChan)
	}()

	results = results[:0]
	done := 0
	failed := 0
	for r := range resChan {
		done++
		results = append(results, r)
		if r.err != nil || r.res == nil {
			failed++
		} else if math.IsNaN(r.res.LogMarginalLik) || math.IsInf(r.res.LogMarginalLik, 0) {
			failed++
		}
		if done%10 == 0 || done == len(siteIDs) {
			log.Printf("    %d/%d sites done (%d failed)", done, len(siteIDs), failed)
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].siteID < results[j].siteID
	})
	return results
}

func writeResults(path string, results []siteResult, hp *inference.HyperParams) {
	paramNames := inference.DefaultSMCConfig().ParamNames
	nParams := len(paramNames)

	header := []string{"SITE_ID", "NUM_YEARS", "LOG_MARGINAL_LIK", "STATUS"}
	for _, name := range paramNames {
		header = append(header, "MEAN_"+name, "STD_"+name)
	}

	f, err := os.Create(path)
	if err != nil {
		log.Fatalf("creating output: %v", err)
	}
	defer f.Close()
	w := csv.NewWriter(f)

	// Write hyperparameters as comment header
	w.Write([]string{fmt.Sprintf(
		"# Hyperparams: mu_flow=%.6f sigma_flow=%.6f mu_temp=%.6f sigma_temp=%.6f mu_do=%.6f sigma_do=%.6f",
		hp.MuBetaFlow, hp.SigmaBetaFlow, hp.MuBetaTemp, hp.SigmaBetaTemp, hp.MuBetaDO, hp.SigmaBetaDO,
	)})
	w.Write(header)

	for _, r := range results {
		row := make([]string, 0, len(header))
		row = append(row, strconv.Itoa(r.siteID), strconv.Itoa(r.numYears))

		if r.err != nil || r.res == nil {
			status := "PANIC"
			if r.err != nil {
				status = r.err.Error()
			}
			row = append(row, "NA", status)
			for range nParams {
				row = append(row, "NA", "NA")
			}
		} else {
			lml := r.res.LogMarginalLik
			if math.IsNaN(lml) || math.IsInf(lml, 0) {
				row = append(row, "NA", "DIVERGED")
				for range nParams {
					row = append(row, "NA", "NA")
				}
			} else {
				row = append(row, fmt.Sprintf("%.4f", lml), "OK")
				for j := range nParams {
					row = append(row,
						fmt.Sprintf("%.6f", r.res.PosteriorMean[j]),
						fmt.Sprintf("%.6f", r.res.PosteriorStd[j]),
					)
				}
			}
		}
		w.Write(row)
	}

	w.Flush()
	if err := w.Error(); err != nil {
		log.Fatalf("writing CSV: %v", err)
	}
}

func logHyperParams(hp *inference.HyperParams) {
	log.Printf("    mu_flow=%.4f  sigma_flow=%.4f", hp.MuBetaFlow, hp.SigmaBetaFlow)
	log.Printf("    mu_temp=%.4f  sigma_temp=%.4f", hp.MuBetaTemp, hp.SigmaBetaTemp)
	log.Printf("    mu_do=%.4f    sigma_do=%.4f", hp.MuBetaDO, hp.SigmaBetaDO)
}
