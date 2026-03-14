package inference

import (
	"math"
	"math/rand/v2"

	"github.com/umbralcalc/anglersim/pkg/data"
)

// GenerateSyntheticData creates a synthetic site time series from known
// Ricker model parameters, for testing parameter recovery.
func GenerateSyntheticData(
	seed uint64,
	T int,
	trueParams []float64, // [r0, alpha, b1, b2, b3, sigma, obs_sd]
) *data.SiteData {
	rng := rand.New(rand.NewPCG(seed, seed+1))

	r0 := trueParams[0]
	alpha := trueParams[1]
	betas := trueParams[2:5]
	sigma := trueParams[5]
	obsSd := trueParams[6]

	// Generate some covariates (standardised)
	covariates := make([][]float64, T)
	for t := range T {
		covariates[t] = []float64{
			rng.NormFloat64() * 0.5, // flow
			rng.NormFloat64() * 0.5, // temp
			rng.NormFloat64() * 0.5, // DO
		}
	}

	// Simulate true population dynamics
	logN := -3.0 // initial log-density
	years := make([]float64, T)
	logDensity := make([][]float64, T)

	for t := range T {
		years[t] = 2000.0 + float64(t)

		// Observe with noise
		observedLogN := logN + rng.NormFloat64()*obsSd
		logDensity[t] = []float64{observedLogN}

		// Advance dynamics (for next step)
		if t < T-1 {
			envEffect := 0.0
			for j := range 3 {
				envEffect += betas[j] * covariates[t+1][j]
			}
			density := math.Exp(logN)
			logN = logN + r0 + envEffect - alpha*density + rng.NormFloat64()*sigma
		}
	}

	return &data.SiteData{
		SiteID:        9999,
		Years:         years,
		LogDensity:    logDensity,
		Covariates:    covariates,
		NumCovariates: 3,
	}
}
