package inference

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/umbralcalc/anglersim/pkg/data"
)

// generateSyntheticData creates a synthetic site time series from known
// Ricker model parameters, for testing parameter recovery.
func generateSyntheticData(
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

func TestRunSMC_SyntheticRecovery(t *testing.T) {
	// True parameters
	trueParams := []float64{
		0.5,  // growth_rate
		2.0,  // density_dependence
		0.0,  // beta_flow (no effect for simpler recovery)
		0.0,  // beta_temp
		0.0,  // beta_do
		0.15, // process_noise_sd
		0.1,  // obs_noise_sd
	}

	d := generateSyntheticData(123, 30, trueParams)

	config := DefaultSMCConfig()
	config.NumParticles = 100
	config.NumRounds = 2
	config.Seed = 456
	config.Verbose = false

	result := RunSMC(d, config)

	if result == nil {
		t.Fatal("RunSMC returned nil")
	}

	// Check that posterior means are in a reasonable range
	// (within 3 posterior SDs of true values for key params)
	t.Run("growth_rate recovery", func(t *testing.T) {
		idx := 0
		diff := math.Abs(result.PosteriorMean[idx] - trueParams[idx])
		tolerance := 3 * result.PosteriorStd[idx]
		if tolerance < 0.5 {
			tolerance = 0.5 // minimum tolerance for noisy estimation
		}
		if diff > tolerance {
			t.Errorf("growth_rate: posterior mean %.4f, true %.4f, diff %.4f > tolerance %.4f",
				result.PosteriorMean[idx], trueParams[idx], diff, tolerance)
		}
	})

	t.Run("density_dependence recovery", func(t *testing.T) {
		idx := 1
		diff := math.Abs(result.PosteriorMean[idx] - trueParams[idx])
		tolerance := 3 * result.PosteriorStd[idx]
		if tolerance < 2.0 {
			tolerance = 2.0 // density_dependence is hard to estimate precisely
		}
		if diff > tolerance {
			t.Errorf("density_dependence: posterior mean %.4f, true %.4f, diff %.4f > tolerance %.4f",
				result.PosteriorMean[idx], trueParams[idx], diff, tolerance)
		}
	})

	t.Run("log_marginal_likelihood finite", func(t *testing.T) {
		if math.IsNaN(result.LogMarginalLik) || math.IsInf(result.LogMarginalLik, 0) {
			t.Errorf("log marginal likelihood is %f", result.LogMarginalLik)
		}
	})

	t.Run("predictions populated", func(t *testing.T) {
		if len(result.Predictions) != len(d.Years) {
			t.Errorf("expected %d prediction rows, got %d",
				len(d.Years), len(result.Predictions))
		}
		if len(result.Predictions) > 0 && len(result.Predictions[0]) != config.NumParticles {
			t.Errorf("expected %d particles per row, got %d",
				config.NumParticles, len(result.Predictions[0]))
		}
	})

	t.Run("posterior_std positive", func(t *testing.T) {
		for i, s := range result.PosteriorStd {
			if s <= 0 || math.IsNaN(s) {
				t.Errorf("param %d: posterior std = %f", i, s)
			}
		}
	})
}

func TestLogSumExp(t *testing.T) {
	// log(exp(1) + exp(2) + exp(3)) = 3 + log(1 + exp(-1) + exp(-2))
	result := logSumExp([]float64{1.0, 2.0, 3.0})
	expected := 3.0 + math.Log(1+math.Exp(-1)+math.Exp(-2))
	if math.Abs(result-expected) > 1e-10 {
		t.Errorf("logSumExp = %.10f, expected %.10f", result, expected)
	}

	// Edge case: all -Inf
	result = logSumExp([]float64{math.Inf(-1), math.Inf(-1)})
	if !math.IsInf(result, -1) {
		t.Errorf("logSumExp of all -Inf should be -Inf, got %f", result)
	}
}

func TestCholeskyDecomp(t *testing.T) {
	// 2x2 identity
	L := choleskyDecomp([]float64{1, 0, 0, 1}, 2)
	if L == nil {
		t.Fatal("cholesky of identity returned nil")
	}
	if math.Abs(L[0]-1) > 1e-10 || math.Abs(L[3]-1) > 1e-10 {
		t.Errorf("cholesky of identity: L = %v", L)
	}

	// Not PD
	L = choleskyDecomp([]float64{1, 0, 0, -1}, 2)
	if L != nil {
		t.Error("expected nil for non-PD matrix")
	}
}

func TestWeightedQuantiles(t *testing.T) {
	values := []float64{1, 2, 3, 4, 5}
	weights := []float64{0.2, 0.2, 0.2, 0.2, 0.2}
	q := WeightedQuantiles(values, weights, []float64{0.5})
	if q[0] != 3 {
		t.Errorf("median = %f, expected 3", q[0])
	}
}
