package inference

import (
	"fmt"

	"github.com/umbralcalc/anglersim/pkg/data"
	"github.com/umbralcalc/anglersim/pkg/population"
	"github.com/umbralcalc/stochadex/pkg/analysis"
	"github.com/umbralcalc/stochadex/pkg/general"
	stdinf "github.com/umbralcalc/stochadex/pkg/inference"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

// SMCConfig holds configuration for the SMC inference run.
type SMCConfig struct {
	NumParticles int
	NumRounds    int
	Seed         uint64
	Priors       []stdinf.Prior
	ParamNames   []string
	Verbose      bool
}

// DefaultSMCConfig returns a default SMC configuration for the Ricker model.
func DefaultSMCConfig() SMCConfig {
	return SMCConfig{
		NumParticles: 500,
		NumRounds:    3,
		Seed:         42,
		Priors:       DefaultRickerPriors(),
		ParamNames: []string{
			"growth_rate", "density_dependence",
			"beta_flow", "beta_temp", "beta_do",
			"process_noise_sd", "obs_noise_var",
			"allee_effect",
		},
	}
}

// buildInnerSimConfig constructs an SMCInnerSimConfig for the Ricker
// model with N parallel particle evaluations.
func buildInnerSimConfig(
	d *data.SiteData,
	N int,
	nParams int,
) *analysis.SMCInnerSimConfig {
	T := len(d.Years)

	partitions := make([]*simulator.PartitionConfig, 0, 2+2*N)
	loglikePartitions := make([]string, N)
	paramForwarding := make(map[string][]int)

	// [0] observed_data
	partitions = append(partitions, &simulator.PartitionConfig{
		Name:      "observed_data",
		Iteration: &general.FromStorageIteration{Data: d.LogDensity},
		Params: simulator.NewParams(
			make(map[string][]float64)),
		InitStateValues:   d.LogDensity[0],
		StateHistoryDepth: 2,
		Seed:              0,
	})

	// [1] covariates
	partitions = append(partitions, &simulator.PartitionConfig{
		Name:      "covariates",
		Iteration: &general.FromStorageIteration{Data: d.Covariates},
		Params: simulator.NewParams(
			make(map[string][]float64)),
		InitStateValues:   d.Covariates[0],
		StateHistoryDepth: 2,
		Seed:              0,
	})

	for p := range N {
		rickerName := fmt.Sprintf("ricker_%d", p)
		llName := fmt.Sprintf("loglike_%d", p)

		// ricker_p
		partitions = append(partitions, &simulator.PartitionConfig{
			Name:      rickerName,
			Iteration: &population.RickerIteration{},
			Params: simulator.NewParams(map[string][]float64{
				"growth_rate":            {0},
				"density_dependence":     {0},
				"covariate_coefficients": {0, 0, 0},
				"covariates":             d.Covariates[0],
				"process_noise_sd":       {0},
				"allee_effect":           {0},
			}),
			ParamsFromUpstream: map[string]simulator.NamedUpstreamConfig{
				"covariates": {Upstream: "covariates"},
			},
			InitStateValues:   d.LogDensity[0],
			StateHistoryDepth: 2,
			Seed:              uint64(p + 1),
		})

		// loglike_p
		partitions = append(partitions, &simulator.PartitionConfig{
			Name: llName,
			Iteration: &stdinf.DataComparisonIteration{
				Likelihood: &stdinf.NormalLikelihoodDistribution{},
			},
			Params: simulator.NewParams(map[string][]float64{
				"mean":               d.LogDensity[0],
				"variance":           {1.0},
				"latest_data_values": d.LogDensity[0],
				"cumulative":         {1},
				"burn_in_steps":      {0},
			}),
			ParamsFromUpstream: map[string]simulator.NamedUpstreamConfig{
				"mean":               {Upstream: rickerName},
				"latest_data_values": {Upstream: "observed_data"},
			},
			InitStateValues:   []float64{0.0},
			StateHistoryDepth: 2,
			Seed:              0,
		})

		loglikePartitions[p] = llName

		// Param forwarding from proposal to inner partitions
		base := p * nParams
		paramForwarding[rickerName+"/growth_rate"] = []int{base + 0}
		paramForwarding[rickerName+"/density_dependence"] = []int{base + 1}
		paramForwarding[rickerName+"/covariate_coefficients"] = []int{base + 2, base + 3, base + 4}
		paramForwarding[rickerName+"/process_noise_sd"] = []int{base + 5}
		paramForwarding[llName+"/variance"] = []int{base + 6}
		paramForwarding[rickerName+"/allee_effect"] = []int{base + 7}
	}

	return &analysis.SMCInnerSimConfig{
		Partitions: partitions,
		Simulation: &simulator.SimulationConfig{
			OutputCondition: &simulator.NilOutputCondition{},
			OutputFunction:  &simulator.NilOutputFunction{},
			TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
				MaxNumberOfSteps: T - 1,
			},
			TimestepFunction: &general.FromStorageTimestepFunction{
				Data: d.Years,
			},
			InitTimeValue: d.Years[0],
		},
		LoglikePartitions: loglikePartitions,
		ParamForwarding:   paramForwarding,
	}
}

// RunSMC runs iterated importance sampling for the Ricker model
// using the stochadex SMC inference tools.
func RunSMC(d *data.SiteData, config SMCConfig) *stdinf.SMCResult {
	model := analysis.SMCParticleModel{
		Build: func(N int, np int) *analysis.SMCInnerSimConfig {
			return buildInnerSimConfig(d, N, np)
		},
	}

	return analysis.RunSMCInference(analysis.AppliedSMCInference{
		ProposalName:  "smc_proposals",
		SimName:       "smc_sim",
		PosteriorName: "smc_posterior",
		NumParticles:  config.NumParticles,
		NumRounds:     config.NumRounds,
		Priors:        config.Priors,
		ParamNames:    config.ParamNames,
		Model:         model,
		Seed:          config.Seed,
		Verbose:       config.Verbose,
	})
}

// RunSMCSafe wraps RunSMC with panic recovery, returning nil and the
// error message if the inner simulation panics.
func RunSMCSafe(d *data.SiteData, config SMCConfig) (result *stdinf.SMCResult, err error) {
	defer func() {
		if r := recover(); r != nil {
			result = nil
			err = fmt.Errorf("%v", r)
		}
	}()
	result = RunSMC(d, config)
	if result == nil {
		return nil, fmt.Errorf("SMC returned nil result")
	}
	return result, nil
}
