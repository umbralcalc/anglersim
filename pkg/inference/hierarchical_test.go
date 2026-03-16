package inference

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestEstimateHyperParams(t *testing.T) {
	t.Run("recovers known population parameters", func(t *testing.T) {
		// True population-level parameters
		trueMuFlow := -0.15
		trueSigmaFlow := 0.10
		trueMuTemp := 0.20
		trueSigmaTemp := 0.08
		trueMuDO := 0.05
		trueSigmaDO := 0.12

		rng := rand.New(rand.NewPCG(42, 43))
		nSites := 200

		sites := make([]SitePosteriorSummary, nSites)
		for i := range nSites {
			// True site-level betas drawn from population
			trueBetaFlow := trueMuFlow + rng.NormFloat64()*trueSigmaFlow
			trueBetaTemp := trueMuTemp + rng.NormFloat64()*trueSigmaTemp
			trueBetaDO := trueMuDO + rng.NormFloat64()*trueSigmaDO

			// Posterior SE (simulating different amounts of data per site)
			seFlow := 0.05 + rng.Float64()*0.15
			seTemp := 0.05 + rng.Float64()*0.15
			seDO := 0.05 + rng.Float64()*0.15

			// Observed posterior mean = true + noise from posterior
			obsFlow := trueBetaFlow + rng.NormFloat64()*seFlow
			obsTemp := trueBetaTemp + rng.NormFloat64()*seTemp
			obsDO := trueBetaDO + rng.NormFloat64()*seDO

			sites[i] = SitePosteriorSummary{
				SiteID:   i,
				NumYears: 10 + int(rng.Float64()*20),
				Mean: []float64{
					0.5,     // growth_rate (not used for hyper)
					5.0,     // density_dependence
					obsFlow, // beta_flow
					obsTemp, // beta_temp
					obsDO,   // beta_do
					0.2,     // process_noise_sd
					0.4,     // obs_noise_var
				},
				Std: []float64{
					0.1,    // growth_rate
					1.0,    // density_dependence
					seFlow, // beta_flow
					seTemp, // beta_temp
					seDO,   // beta_do
					0.05,   // process_noise_sd
					0.1,    // obs_noise_var
				},
				LogMargLik: -10.0,
			}
		}

		hp := EstimateHyperParams(sites)

		// Check recovery within reasonable tolerance
		// With 200 sites, we should get close
		tol := 0.05
		if math.Abs(hp.MuBetaFlow-trueMuFlow) > tol {
			t.Errorf("MuBetaFlow: got %.4f, want %.4f (±%.2f)", hp.MuBetaFlow, trueMuFlow, tol)
		}
		if math.Abs(hp.MuBetaTemp-trueMuTemp) > tol {
			t.Errorf("MuBetaTemp: got %.4f, want %.4f (±%.2f)", hp.MuBetaTemp, trueMuTemp, tol)
		}
		if math.Abs(hp.MuBetaDO-trueMuDO) > tol {
			t.Errorf("MuBetaDO: got %.4f, want %.4f (±%.2f)", hp.MuBetaDO, trueMuDO, tol)
		}

		// Sigma recovery is noisier but should be in the right ballpark
		sigmaTol := 0.08
		if math.Abs(hp.SigmaBetaFlow-trueSigmaFlow) > sigmaTol {
			t.Errorf("SigmaBetaFlow: got %.4f, want %.4f (±%.2f)", hp.SigmaBetaFlow, trueSigmaFlow, sigmaTol)
		}
		if math.Abs(hp.SigmaBetaTemp-trueSigmaTemp) > sigmaTol {
			t.Errorf("SigmaBetaTemp: got %.4f, want %.4f (±%.2f)", hp.SigmaBetaTemp, trueSigmaTemp, sigmaTol)
		}
		if math.Abs(hp.SigmaBetaDO-trueSigmaDO) > sigmaTol {
			t.Errorf("SigmaBetaDO: got %.4f, want %.4f (±%.2f)", hp.SigmaBetaDO, trueSigmaDO, sigmaTol)
		}

		t.Logf("Recovered: mu_flow=%.4f (true=%.2f), sigma_flow=%.4f (true=%.2f)",
			hp.MuBetaFlow, trueMuFlow, hp.SigmaBetaFlow, trueSigmaFlow)
		t.Logf("Recovered: mu_temp=%.4f (true=%.2f), sigma_temp=%.4f (true=%.2f)",
			hp.MuBetaTemp, trueMuTemp, hp.SigmaBetaTemp, trueSigmaTemp)
		t.Logf("Recovered: mu_do=%.4f (true=%.2f), sigma_do=%.4f (true=%.2f)",
			hp.MuBetaDO, trueMuDO, hp.SigmaBetaDO, trueSigmaDO)
	})

	t.Run("handles zero between-site variance", func(t *testing.T) {
		// All sites have the same true beta — sigma should be near the floor
		sites := make([]SitePosteriorSummary, 50)
		for i := range 50 {
			sites[i] = SitePosteriorSummary{
				SiteID:   i,
				NumYears: 20,
				Mean:     []float64{0.5, 5.0, 0.1, 0.1, 0.1, 0.2, 0.4},
				Std:      []float64{0.1, 1.0, 0.02, 0.02, 0.02, 0.05, 0.1},
				LogMargLik: -10.0,
			}
		}

		hp := EstimateHyperParams(sites)

		// Sigma should be at the floor (0.01)
		if hp.SigmaBetaFlow > 0.05 {
			t.Errorf("SigmaBetaFlow should be near floor, got %.4f", hp.SigmaBetaFlow)
		}
		// Mu should be close to the common value
		if math.Abs(hp.MuBetaFlow-0.1) > 0.02 {
			t.Errorf("MuBetaFlow: got %.4f, want ~0.1", hp.MuBetaFlow)
		}
	})
}

func TestHierarchicalPriors(t *testing.T) {
	t.Run("updates only covariate priors", func(t *testing.T) {
		hp := &HyperParams{
			MuBetaFlow:    -0.15,
			MuBetaTemp:    0.20,
			MuBetaDO:      0.05,
			SigmaBetaFlow: 0.10,
			SigmaBetaTemp: 0.08,
			SigmaBetaDO:   0.12,
		}

		priors := HierarchicalPriors(hp)
		defaults := DefaultRickerPriors()

		// Check non-covariate priors are unchanged
		// growth_rate (index 0)
		if priors[0].(*TruncatedNormalPrior).Mu != defaults[0].(*TruncatedNormalPrior).Mu {
			t.Error("growth_rate prior should be unchanged")
		}
		// density_dependence (index 1)
		if priors[1].(*LogNormalPrior).Mu != defaults[1].(*LogNormalPrior).Mu {
			t.Error("density_dependence prior should be unchanged")
		}
		// process_noise_sd (index 5)
		if priors[5].(*HalfNormalPrior).Sigma != defaults[5].(*HalfNormalPrior).Sigma {
			t.Error("process_noise_sd prior should be unchanged")
		}
		// obs_noise_var (index 6)
		if priors[6].(*LogNormalPrior).Mu != defaults[6].(*LogNormalPrior).Mu {
			t.Error("obs_noise_var prior should be unchanged")
		}

		// Check covariate priors are updated
		flowPrior := priors[2].(*TruncatedNormalPrior)
		if flowPrior.Mu != hp.MuBetaFlow || flowPrior.Sigma != hp.SigmaBetaFlow {
			t.Errorf("beta_flow prior: got mu=%.4f sigma=%.4f, want mu=%.4f sigma=%.4f",
				flowPrior.Mu, flowPrior.Sigma, hp.MuBetaFlow, hp.SigmaBetaFlow)
		}
		tempPrior := priors[3].(*TruncatedNormalPrior)
		if tempPrior.Mu != hp.MuBetaTemp || tempPrior.Sigma != hp.SigmaBetaTemp {
			t.Errorf("beta_temp prior: got mu=%.4f sigma=%.4f, want mu=%.4f sigma=%.4f",
				tempPrior.Mu, tempPrior.Sigma, hp.MuBetaTemp, hp.SigmaBetaTemp)
		}
		doPrior := priors[4].(*TruncatedNormalPrior)
		if doPrior.Mu != hp.MuBetaDO || doPrior.Sigma != hp.SigmaBetaDO {
			t.Errorf("beta_do prior: got mu=%.4f sigma=%.4f, want mu=%.4f sigma=%.4f",
				doPrior.Mu, doPrior.Sigma, hp.MuBetaDO, hp.SigmaBetaDO)
		}
	})
}
