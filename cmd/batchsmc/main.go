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
	panelFile := flag.String("panel", "dat/brown_trout_panel_with_covariates.csv",
		"panel CSV with covariates")
	outFile := flag.String("out", "dat/batch_smc_results.csv", "output CSV path")
	workers := flag.Int("workers", 4, "number of parallel workers")
	particles := flag.Int("particles", 500, "particles per SMC run")
	rounds := flag.Int("rounds", 3, "SMC importance sampling rounds")
	baseSeed := flag.Uint64("seed", 42, "base random seed")
	minYears := flag.Int("min-years", 10, "minimum years of data per site")
	verbose := flag.Bool("verbose", false, "verbose per-site output")
	flag.Parse()

	// Load all sites in one pass
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
	log.Printf("Fitting %d sites with >= %d years (N=%d, rounds=%d, workers=%d)",
		len(siteIDs), *minYears, *particles, *rounds, *workers)

	// Worker pool
	type job struct {
		siteID int
		sd     *data.SiteData
	}
	type result struct {
		siteID   int
		numYears int
		res      *inference.SMCResult
		err      error
	}

	jobs := make(chan job, len(siteIDs))
	results := make(chan result, len(siteIDs))

	var wg sync.WaitGroup
	for w := 0; w < *workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				cfg := inference.DefaultSMCConfig()
				cfg.NumParticles = *particles
				cfg.NumRounds = *rounds
				cfg.Seed = *baseSeed + uint64(j.siteID)
				cfg.Verbose = *verbose
				r, err := inference.RunSMCSafe(j.sd, cfg)
				results <- result{
					siteID:   j.siteID,
					numYears: len(j.sd.Years),
					res:      r,
					err:      err,
				}
			}
		}()
	}

	// Send jobs
	for _, id := range siteIDs {
		jobs <- job{siteID: id, sd: allSites[id]}
	}
	close(jobs)

	// Collect results
	go func() {
		wg.Wait()
		close(results)
	}()

	paramNames := inference.DefaultSMCConfig().ParamNames
	nParams := len(paramNames)

	// Build CSV header
	header := []string{"SITE_ID", "NUM_YEARS", "LOG_MARGINAL_LIK", "STATUS"}
	for _, name := range paramNames {
		header = append(header, "MEAN_"+name, "STD_"+name)
	}

	var rows [][]string
	done := 0
	failed := 0
	for r := range results {
		done++
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
			failed++
		} else {
			lml := r.res.LogMarginalLik
			if math.IsNaN(lml) || math.IsInf(lml, 0) {
				row = append(row, "NA", "DIVERGED")
				for range nParams {
					row = append(row, "NA", "NA")
				}
				failed++
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
		rows = append(rows, row)

		if done%10 == 0 || done == len(siteIDs) {
			log.Printf("  %d/%d sites done (%d failed)", done, len(siteIDs), failed)
		}
	}

	// Sort rows by site ID for deterministic output
	sort.Slice(rows, func(i, j int) bool {
		a, _ := strconv.Atoi(rows[i][0])
		b, _ := strconv.Atoi(rows[j][0])
		return a < b
	})

	// Write CSV
	f, err := os.Create(*outFile)
	if err != nil {
		log.Fatalf("creating output: %v", err)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	w.Write(header)
	for _, row := range rows {
		w.Write(row)
	}
	w.Flush()
	if err := w.Error(); err != nil {
		log.Fatalf("writing CSV: %v", err)
	}

	log.Printf("Done. %d/%d sites fitted successfully. Results in %s",
		done-failed, done, *outFile)
}
