package inference

import (
	"math"
	"testing"

	"github.com/umbralcalc/anglersim/pkg/data"
	"github.com/umbralcalc/stochadex/pkg/analysis"
	"github.com/umbralcalc/stochadex/pkg/general"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

// generateSyntheticData creates a synthetic site time series from known
// Ricker model parameters, for testing parameter recovery.
func generateSyntheticData(
	seed uint64,
	T int,
	trueParams []float64,
) *data.SiteData {
	return GenerateSyntheticData(seed, T, trueParams)
}

func TestSMCDecomposed_Harness(t *testing.T) {
	t.Run(
		"test that decomposed SMC iterations run with harnesses",
		func(t *testing.T) {
			trueParams := []float64{
				0.5, 2.0, 0.0, 0.0, 0.0, 0.15, 0.01,
			}
			siteData := generateSyntheticData(123, 15, trueParams)
			numParticles := 10
			nParams := 7

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

func TestStateSliceIteration_Harness(t *testing.T) {
	t.Run(
		"test that StateSliceIteration runs with harnesses",
		func(t *testing.T) {
			settings := &simulator.Settings{
				Iterations: []simulator.IterationSettings{
					{
						Name: "upstream",
						Params: simulator.NewParams(map[string][]float64{
							"param_values": {1.0, 2.0, 3.0, 4.0, 5.0},
						}),
						InitStateValues:   []float64{1.0, 2.0, 3.0, 4.0, 5.0},
						StateWidth:        5,
						StateHistoryDepth: 2,
						Seed:              0,
					},
					{
						Name: "slice",
						Params: simulator.NewParams(map[string][]float64{
							"offset":        {1},
							"width":         {3},
							"latest_values": {1.0, 2.0, 3.0, 4.0, 5.0},
						}),
						ParamsFromUpstream: map[string]simulator.UpstreamConfig{
							"latest_values": {Upstream: 0},
						},
						InitStateValues:   []float64{2.0, 3.0, 4.0},
						StateWidth:        3,
						StateHistoryDepth: 1,
						Seed:              0,
					},
				},
				InitTimeValue:         0.0,
				TimestepsHistoryDepth: 2,
			}
			settings.Init()

			iterations := []simulator.Iteration{
				&general.ParamValuesIteration{},
				&StateSliceIteration{},
			}
			implementations := &simulator.Implementations{
				Iterations:      iterations,
				OutputCondition: &simulator.EveryStepOutputCondition{},
				OutputFunction:  &simulator.NilOutputFunction{},
				TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
					MaxNumberOfSteps: 3,
				},
				TimestepFunction: &simulator.ConstantTimestepFunction{Stepsize: 1.0},
			}
			if err := simulator.RunWithHarnesses(settings, implementations); err != nil {
				t.Errorf("test harness failed: %v", err)
			}
		},
	)
}

func TestLogSumExp(t *testing.T) {
	result := logSumExp([]float64{1.0, 2.0, 3.0})
	expected := 3.0 + math.Log(1+math.Exp(-1)+math.Exp(-2))
	if math.Abs(result-expected) > 1e-10 {
		t.Errorf("logSumExp = %.10f, expected %.10f", result, expected)
	}

	result = logSumExp([]float64{math.Inf(-1), math.Inf(-1)})
	if !math.IsInf(result, -1) {
		t.Errorf("logSumExp of all -Inf should be -Inf, got %f", result)
	}
}

func TestCholeskyDecomp(t *testing.T) {
	L := choleskyDecomp([]float64{1, 0, 0, 1}, 2)
	if L == nil {
		t.Fatal("cholesky of identity returned nil")
	}
	if math.Abs(L[0]-1) > 1e-10 || math.Abs(L[3]-1) > 1e-10 {
		t.Errorf("cholesky of identity: L = %v", L)
	}

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
