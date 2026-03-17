package inference

import (
	"encoding/csv"
	"math"
	"os"
	"strconv"

	stdinf "github.com/umbralcalc/stochadex/pkg/inference"
)

// SitePosteriorSummary holds the Stage 1 posterior for one site.
type SitePosteriorSummary struct {
	SiteID     int
	NumYears   int
	Mean       []float64 // [d] posterior means
	Std        []float64 // [d] posterior stds
	LogMargLik float64
}

// HyperParams holds population-level hyperparameters for covariate effects.
type HyperParams struct {
	MuBetaFlow    float64
	MuBetaTemp    float64
	MuBetaDO      float64
	SigmaBetaFlow float64
	SigmaBetaTemp float64
	SigmaBetaDO   float64
}

// LoadBatchResults reads the CSV produced by cmd/batchsmc and returns
// site posterior summaries for OK sites.
func LoadBatchResults(path string) []SitePosteriorSummary {
	f, err := os.Open(path)
	if err != nil {
		panic("opening batch results: " + err.Error())
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.Read() // skip header
	records, err := r.ReadAll()
	if err != nil {
		panic("reading batch results: " + err.Error())
	}

	nParams := 7
	var sites []SitePosteriorSummary
	for _, rec := range records {
		if rec[3] != "OK" {
			continue
		}
		id, _ := strconv.Atoi(rec[0])
		ny, _ := strconv.Atoi(rec[1])
		lml, _ := strconv.ParseFloat(rec[2], 64)
		means := make([]float64, nParams)
		stds := make([]float64, nParams)
		for i := range nParams {
			means[i], _ = strconv.ParseFloat(rec[4+2*i], 64)
			stds[i], _ = strconv.ParseFloat(rec[5+2*i], 64)
		}
		sites = append(sites, SitePosteriorSummary{
			SiteID:     id,
			NumYears:   ny,
			Mean:       means,
			Std:        stds,
			LogMargLik: lml,
		})
	}
	return sites
}

// EstimateHyperParams estimates population-level hyperparameters from
// independent site-level posterior summaries using empirical Bayes
// (maximum marginal likelihood).
//
// For each covariate effect j (indices 2,3,4 = beta_flow, beta_temp, beta_do):
//
//	mu_j, sigma_j = argmax sum_s log N(mean_s[j] | mu, sigma^2 + std_s[j]^2)
//
// The MLE for mu given sigma is a precision-weighted average.
// The MLE for sigma is found by 1D grid search on the profile likelihood.
func EstimateHyperParams(sites []SitePosteriorSummary) *HyperParams {
	hp := &HyperParams{}

	hp.MuBetaFlow, hp.SigmaBetaFlow = estimateNormalHyper(sites, 2)
	hp.MuBetaTemp, hp.SigmaBetaTemp = estimateNormalHyper(sites, 3)
	hp.MuBetaDO, hp.SigmaBetaDO = estimateNormalHyper(sites, 4)

	return hp
}

// estimateNormalHyper estimates mu and sigma for a single parameter index
// using the normal-normal marginal likelihood model:
//
//	hat_beta_s ~ N(mu, sigma^2 + se_s^2)
//
// Returns (mu, sigma).
func estimateNormalHyper(sites []SitePosteriorSummary, paramIdx int) (float64, float64) {
	n := len(sites)
	if n == 0 {
		return 0, 1
	}

	// Extract site means and squared standard errors
	betaHats := make([]float64, n)
	se2s := make([]float64, n)
	for i, s := range sites {
		betaHats[i] = s.Mean[paramIdx]
		se2s[i] = s.Std[paramIdx] * s.Std[paramIdx]
	}

	// Grid search over sigma in [0, maxSigma]
	// maxSigma is the empirical std of betaHats (an upper bound)
	empiricalMean := 0.0
	for _, b := range betaHats {
		empiricalMean += b
	}
	empiricalMean /= float64(n)

	empiricalVar := 0.0
	for _, b := range betaHats {
		d := b - empiricalMean
		empiricalVar += d * d
	}
	empiricalVar /= float64(n)
	maxSigma := math.Sqrt(empiricalVar) + 0.01

	nGrid := 200
	bestSigma := 0.0
	bestLogLik := math.Inf(-1)

	for g := range nGrid + 1 {
		sigma := maxSigma * float64(g) / float64(nGrid)
		sigma2 := sigma * sigma

		// MLE for mu given sigma: precision-weighted mean
		muNum := 0.0
		muDen := 0.0
		for i := range n {
			totalVar := sigma2 + se2s[i]
			if totalVar < 1e-15 {
				totalVar = 1e-15
			}
			prec := 1.0 / totalVar
			muNum += prec * betaHats[i]
			muDen += prec
		}
		mu := muNum / muDen

		// Profile log-likelihood
		ll := 0.0
		for i := range n {
			totalVar := sigma2 + se2s[i]
			if totalVar < 1e-15 {
				totalVar = 1e-15
			}
			d := betaHats[i] - mu
			ll += -0.5*math.Log(2*math.Pi*totalVar) - 0.5*d*d/totalVar
		}

		if ll > bestLogLik {
			bestLogLik = ll
			bestSigma = sigma
		}
	}

	// Compute final mu at best sigma
	sigma2 := bestSigma * bestSigma
	muNum := 0.0
	muDen := 0.0
	for i := range n {
		totalVar := sigma2 + se2s[i]
		if totalVar < 1e-15 {
			totalVar = 1e-15
		}
		prec := 1.0 / totalVar
		muNum += prec * betaHats[i]
		muDen += prec
	}
	bestMu := muNum / muDen

	// Floor sigma to avoid degenerate priors
	if bestSigma < 0.01 {
		bestSigma = 0.01
	}

	return bestMu, bestSigma
}

// HierarchicalPriors returns updated priors for the Ricker model where
// the covariate effects (beta_flow, beta_temp, beta_do) use informative
// priors derived from population-level hyperparameters. All other priors
// remain at their defaults.
func HierarchicalPriors(hp *HyperParams) []stdinf.Prior {
	priors := DefaultRickerPriors()
	priors[2] = &stdinf.TruncatedNormalPrior{Mu: hp.MuBetaFlow, Sigma: hp.SigmaBetaFlow, Lo: -2.0, Hi: 2.0}
	priors[3] = &stdinf.TruncatedNormalPrior{Mu: hp.MuBetaTemp, Sigma: hp.SigmaBetaTemp, Lo: -2.0, Hi: 2.0}
	priors[4] = &stdinf.TruncatedNormalPrior{Mu: hp.MuBetaDO, Sigma: hp.SigmaBetaDO, Lo: -2.0, Hi: 2.0}
	return priors
}
