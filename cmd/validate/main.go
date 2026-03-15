package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"sort"
	"strconv"
	"sync"

	"github.com/umbralcalc/anglersim/pkg/data"
	"github.com/umbralcalc/anglersim/pkg/inference"
)

type prediction struct {
	siteID int
	year   float64
	obs    float64
	mean   float64
	lo90   float64
	hi90   float64
	lo50   float64
	hi50   float64
}

type result struct {
	siteID     int
	numTrain   int
	numTest    int
	trainLogZ  float64
	rmse       float64
	mae        float64
	coverage90 float64
	coverage50 float64
	status     string
	preds      []prediction
}

func main() {
	panelFile := flag.String("panel", "dat/brown_trout_panel_with_covariates.csv",
		"panel CSV with covariates")
	outFile := flag.String("out", "dat/validation_results.csv", "output CSV path")
	predsFile := flag.String("preds", "dat/validation_predictions.csv",
		"per-year predictions CSV path")
	workers := flag.Int("workers", 4, "number of parallel workers")
	particles := flag.Int("particles", 500, "particles per SMC run")
	rounds := flag.Int("rounds", 3, "SMC importance sampling rounds")
	baseSeed := flag.Uint64("seed", 42, "base random seed")
	minYears := flag.Int("min-years", 12, "minimum total years (train+test)")
	holdout := flag.Int("holdout", 3, "number of years to hold out")
	nSims := flag.Int("sims", 200, "forward simulation replicates for prediction intervals")
	verbose := flag.Bool("verbose", false, "verbose per-site output")
	flag.Parse()

	log.Printf("Loading sites from %s ...", *panelFile)
	allSites := data.LoadAllSiteTimeSeries(*panelFile)
	log.Printf("Loaded %d sites total", len(allSites))

	var siteIDs []int
	for id, sd := range allSites {
		if len(sd.Years) >= *minYears {
			siteIDs = append(siteIDs, id)
		}
	}
	sort.Ints(siteIDs)
	log.Printf("Validating %d sites with >= %d years (holdout=%d, N=%d, rounds=%d, sims=%d)",
		len(siteIDs), *minYears, *holdout, *particles, *rounds, *nSims)

	type job struct {
		siteID int
		sd     *data.SiteData
	}

	jobs := make(chan job, len(siteIDs))
	results := make(chan result, len(siteIDs))

	var wg sync.WaitGroup
	for w := 0; w < *workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				results <- runValidation(j.siteID, j.sd, *holdout, *particles,
					*rounds, *baseSeed, *nSims, *verbose)
			}
		}()
	}

	for _, id := range siteIDs {
		jobs <- job{siteID: id, sd: allSites[id]}
	}
	close(jobs)

	go func() {
		wg.Wait()
		close(results)
	}()

	var rows []result
	done := 0
	failed := 0
	for r := range results {
		done++
		if r.status != "OK" {
			failed++
		}
		rows = append(rows, r)
		if done%10 == 0 || done == len(siteIDs) {
			log.Printf("  %d/%d sites done (%d failed)", done, len(siteIDs), failed)
		}
	}

	sort.Slice(rows, func(i, j int) bool {
		return rows[i].siteID < rows[j].siteID
	})

	// Write summary CSV
	f, err := os.Create(*outFile)
	if err != nil {
		log.Fatalf("creating output: %v", err)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	w.Write([]string{"SITE_ID", "NUM_TRAIN", "NUM_TEST", "TRAIN_LOG_MARGINAL_LIK",
		"RMSE", "MAE", "COVERAGE_90", "COVERAGE_50", "STATUS"})
	for _, r := range rows {
		row := []string{
			strconv.Itoa(r.siteID),
			strconv.Itoa(r.numTrain),
			strconv.Itoa(r.numTest),
		}
		if r.status == "OK" {
			row = append(row,
				fmt.Sprintf("%.4f", r.trainLogZ),
				fmt.Sprintf("%.4f", r.rmse),
				fmt.Sprintf("%.4f", r.mae),
				fmt.Sprintf("%.4f", r.coverage90),
				fmt.Sprintf("%.4f", r.coverage50),
				"OK",
			)
		} else {
			row = append(row, "NA", "NA", "NA", "NA", "NA", r.status)
		}
		w.Write(row)
	}
	w.Flush()

	// Write predictions CSV
	pf, err := os.Create(*predsFile)
	if err != nil {
		log.Fatalf("creating predictions file: %v", err)
	}
	defer pf.Close()
	pw := csv.NewWriter(pf)
	pw.Write([]string{"SITE_ID", "YEAR", "OBSERVED", "PRED_MEAN",
		"PRED_LO90", "PRED_HI90", "PRED_LO50", "PRED_HI50"})
	for _, r := range rows {
		for _, p := range r.preds {
			pw.Write([]string{
				strconv.Itoa(p.siteID),
				fmt.Sprintf("%.0f", p.year),
				fmt.Sprintf("%.6f", p.obs),
				fmt.Sprintf("%.6f", p.mean),
				fmt.Sprintf("%.6f", p.lo90),
				fmt.Sprintf("%.6f", p.hi90),
				fmt.Sprintf("%.6f", p.lo50),
				fmt.Sprintf("%.6f", p.hi50),
			})
		}
	}
	pw.Flush()

	log.Printf("Done. %d/%d sites validated. Results: %s, Predictions: %s",
		done-failed, done, *outFile, *predsFile)
}

func runValidation(siteID int, sd *data.SiteData, holdout, particles,
	rounds int, baseSeed uint64, nSims int, verbose bool,
) result {
	r := result{siteID: siteID}

	defer func() {
		if rec := recover(); rec != nil {
			r.status = fmt.Sprintf("%v", rec)
		}
	}()

	train, test := data.TruncateSiteData(sd, holdout)
	r.numTrain = len(train.Years)
	r.numTest = len(test.Years)

	// Fit on training data
	cfg := inference.DefaultSMCConfig()
	cfg.NumParticles = particles
	cfg.NumRounds = rounds
	cfg.Seed = baseSeed + uint64(siteID)
	cfg.Verbose = verbose

	smcResult, err := inference.RunSMCSafe(train, cfg)
	if err != nil || smcResult == nil {
		r.status = "FIT_FAILED"
		if err != nil {
			r.status = err.Error()
		}
		return r
	}

	lml := smcResult.LogMarginalLik
	if math.IsNaN(lml) || math.IsInf(lml, 0) {
		r.status = "DIVERGED"
		return r
	}
	r.trainLogZ = lml

	// Forward simulate from last training observation using posterior particles
	lastTrainLogDens := train.LogDensity[len(train.LogDensity)-1][0]
	K := len(test.Years)

	rng := rand.New(rand.NewPCG(baseSeed+uint64(siteID)+1000, 0))

	// Simulate nSims trajectories
	trajectories := make([][]float64, nSims)
	for s := range nSims {
		// Sample a particle proportional to weights
		pIdx := weightedSample(smcResult.Weights, rng)
		pp := smcResult.ParticleParams[pIdx]

		r0 := pp[0]
		alpha := pp[1]
		betas := pp[2:5]
		sigma := pp[5]
		obsVar := pp[6]
		obsSd := math.Sqrt(obsVar)

		traj := make([]float64, K)
		logN := lastTrainLogDens
		for t := range K {
			envEffect := 0.0
			for k := 0; k < 3 && k < len(test.Covariates[t]); k++ {
				envEffect += betas[k] * test.Covariates[t][k]
			}
			logN = logN + r0 + envEffect - alpha*math.Exp(logN) + rng.NormFloat64()*sigma
			// Clip to prevent divergence (log-density beyond ±20 is unphysical)
			if logN > 20 {
				logN = 20
			} else if logN < -20 {
				logN = -20
			}
			// Add observation noise — held-out data includes both process and obs noise
			traj[t] = logN + rng.NormFloat64()*obsSd
		}
		trajectories[s] = traj
	}

	// Compute prediction statistics per test year
	var sumSqErr, sumAbsErr float64
	in90, in50 := 0, 0

	for t := range K {
		obs := test.LogDensity[t][0]

		vals := make([]float64, nSims)
		for s := range nSims {
			vals[s] = trajectories[s][t]
		}
		sort.Float64s(vals)

		predMean := 0.0
		for _, v := range vals {
			predMean += v
		}
		predMean /= float64(nSims)

		lo90 := vals[int(float64(nSims)*0.05)]
		hi90 := vals[int(float64(nSims)*0.95)]
		lo50 := vals[int(float64(nSims)*0.25)]
		hi50 := vals[int(float64(nSims)*0.75)]

		residual := obs - predMean
		sumSqErr += residual * residual
		sumAbsErr += math.Abs(residual)

		if obs >= lo90 && obs <= hi90 {
			in90++
		}
		if obs >= lo50 && obs <= hi50 {
			in50++
		}

		r.preds = append(r.preds, prediction{
			siteID: siteID,
			year:   test.Years[t],
			obs:    obs,
			mean:   predMean,
			lo90:   lo90,
			hi90:   hi90,
			lo50:   lo50,
			hi50:   hi50,
		})
	}

	r.rmse = math.Sqrt(sumSqErr / float64(K))
	r.mae = sumAbsErr / float64(K)
	r.coverage90 = float64(in90) / float64(K)
	r.coverage50 = float64(in50) / float64(K)
	r.status = "OK"
	return r
}

func weightedSample(weights []float64, rng *rand.Rand) int {
	u := rng.Float64()
	cum := 0.0
	for i, w := range weights {
		cum += w
		if u <= cum {
			return i
		}
	}
	return len(weights) - 1
}
