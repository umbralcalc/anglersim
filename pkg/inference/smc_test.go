package inference

import (
	"math"
	"testing"

	"github.com/umbralcalc/stochadex/pkg/analysis"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

func TestSMCDecomposed_Harness(t *testing.T) {
	t.Run(
		"test that decomposed SMC iterations run with harnesses",
		func(t *testing.T) {
			trueParams := []float64{
				0.5, 2.0, 0.0, 0.0, 0.0, 0.15, 0.01, 50.0,
			}
			siteData := GenerateSyntheticData(123, 15, trueParams)
			numParticles := 10
			nParams := 8

			applied := analysis.AppliedSMCInference{
				ProposalName:  "smc_proposals",
				SimName:       "smc_sim",
				PosteriorName: "smc_posterior",
				NumParticles:  numParticles,
				NumRounds:     2,
				Priors:        DefaultRickerPriors(),
				ParamNames:    DefaultSMCConfig().ParamNames,
				Model: analysis.SMCParticleModel{
					Build: func(N int, np int) *analysis.SMCInnerSimConfig {
						return buildInnerSimConfig(siteData, N, np)
					},
				},
				Seed:    42,
				Verbose: false,
			}

			partitions := analysis.NewSMCInferencePartitions(applied)
			storage := analysis.NewStateTimeStorageFromPartitions(
				partitions,
				&simulator.NumberOfStepsTerminationCondition{
					MaxNumberOfSteps: 2,
				},
				&simulator.ConstantTimestepFunction{Stepsize: 1.0},
				0.0,
			)

			// Verify output has data
			proposalVals := storage.GetValues("smc_proposals")
			if len(proposalVals) == 0 {
				t.Fatal("no proposal output stored")
			}
			// Verify state widths are correct
			if len(proposalVals[0]) != numParticles*nParams {
				t.Errorf("proposal state width=%d, expected %d",
					len(proposalVals[0]), numParticles*nParams)
			}
		},
	)
}

func TestRunSMC_SyntheticRecovery(t *testing.T) {
	trueParams := []float64{
		0.5,  // growth_rate
		2.0,  // density_dependence
		0.0,  // beta_flow
		0.0,  // beta_temp
		0.0,  // beta_do
		0.15, // process_noise_sd
		0.01, // obs_noise_var
		50.0, // allee_effect
	}

	d := GenerateSyntheticData(123, 30, trueParams)

	config := DefaultSMCConfig()
	config.NumParticles = 100
	config.NumRounds = 2
	config.Seed = 456
	config.Verbose = false

	result := RunSMC(d, config)

	if result == nil {
		t.Fatal("RunSMC returned nil")
	}

	t.Run("growth_rate recovery", func(t *testing.T) {
		idx := 0
		diff := math.Abs(result.PosteriorMean[idx] - trueParams[idx])
		tolerance := 3 * result.PosteriorStd[idx]
		if tolerance < 0.5 {
			tolerance = 0.5
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
			tolerance = 2.0
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

	t.Run("posterior_std positive", func(t *testing.T) {
		for i, s := range result.PosteriorStd {
			if s <= 0 || math.IsNaN(s) {
				t.Errorf("param %d: posterior std = %f", i, s)
			}
		}
	})
}
