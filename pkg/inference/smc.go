package inference

import (
	"fmt"

	"github.com/umbralcalc/anglersim/pkg/data"
	"github.com/umbralcalc/anglersim/pkg/population"
	"github.com/umbralcalc/stochadex/pkg/general"
	stdinf "github.com/umbralcalc/stochadex/pkg/inference"
	"github.com/umbralcalc/stochadex/pkg/macros"
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

// buildInnerSimConfig constructs one particle's Ricker model. The SMC macro
// instantiates it per particle and seeds each independently.
func buildInnerSimConfig(
	d *data.SiteData,
	nParams int,
) *macros.SMCInnerSimConfig {
	T := len(d.Years)

	partitions := make([]*simulator.PartitionConfig, 0, 4)
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

	const rickerName = "ricker"
	const llName = "loglike"

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
		Seed:              1,
	})

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

	// Indices into this particle's own parameter vector.
	paramForwarding[rickerName+"/growth_rate"] = []int{0}
	paramForwarding[rickerName+"/density_dependence"] = []int{1}
	paramForwarding[rickerName+"/covariate_coefficients"] = []int{2, 3, 4}
	paramForwarding[rickerName+"/process_noise_sd"] = []int{5}
	paramForwarding[llName+"/variance"] = []int{6}
	paramForwarding[rickerName+"/allee_effect"] = []int{7}
	_ = nParams

	return &macros.SMCInnerSimConfig{
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
		LoglikePartition: llName,
		ParamForwarding:  paramForwarding,
	}
}

// RunSMC runs iterated importance sampling for the Ricker model
// using the stochadex SMC inference tools.
func RunSMC(d *data.SiteData, config SMCConfig) *stdinf.SMCResult {
	model := macros.SMCParticleModel{
		Build: func(np int) *macros.SMCInnerSimConfig {
			return buildInnerSimConfig(d, np)
		},
	}

	return macros.RunSMCInference(macros.AppliedSMCInference{
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
