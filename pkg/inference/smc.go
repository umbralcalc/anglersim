package inference

import (
	"fmt"
	"math"

	"github.com/umbralcalc/anglersim/pkg/data"
	"github.com/umbralcalc/anglersim/pkg/population"
	"github.com/umbralcalc/stochadex/pkg/analysis"
	"github.com/umbralcalc/stochadex/pkg/general"
	stdinf "github.com/umbralcalc/stochadex/pkg/inference"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

// Re-export SMC types from stochadex.
type SMCResult = stdinf.SMCResult

// WeightedQuantiles computes weighted quantiles from particle values.
var WeightedQuantiles = stdinf.WeightedQuantiles

// SMCConfig holds configuration for the SMC inference run.
type SMCConfig struct {
	NumParticles int
	NumRounds    int
	Seed         uint64
	Priors       []Prior
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
		},
	}
}

// NewNormalDataComparison returns a DataComparisonIteration with a
// NormalLikelihoodDistribution.
func NewNormalDataComparison() simulator.Iteration {
	return &stdinf.DataComparisonIteration{
		Likelihood: &stdinf.NormalLikelihoodDistribution{},
	}
}

// StateSliceIteration extracts a contiguous slice from an upstream
// partition's state.
type StateSliceIteration struct{}

func (s *StateSliceIteration) Configure(
	partitionIndex int,
	settings *simulator.Settings,
) {
}

func (s *StateSliceIteration) Iterate(
	params *simulator.Params,
	partitionIndex int,
	stateHistories []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) []float64 {
	values := params.Get("latest_values")
	offset := int(params.GetIndex("offset", 0))
	width := int(params.GetIndex("width", 0))
	result := make([]float64, width)
	copy(result, values[offset:offset+width])
	return result
}

// PosteriorStateWidth returns the state width for the SMC posterior.
var PosteriorStateWidth = stdinf.PosteriorStateWidth

// EmbeddedStateWidth returns the state width for the embedded inner
// simulation's concatenated output.
func EmbeddedStateWidth(numParticles, numCovariates int) int {
	return 1 + numCovariates + 2*numParticles
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
			Name: rickerName,
			Iteration: &population.RickerIteration{},
			Params: simulator.NewParams(map[string][]float64{
				"growth_rate":            {0},
				"density_dependence":     {0},
				"covariate_coefficients": {0, 0, 0},
				"covariates":             d.Covariates[0],
				"process_noise_sd":       {0},
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
	}

	return &analysis.SMCInnerSimConfig{
		Partitions: partitions,
		Simulation: &simulator.SimulationConfig{
			OutputCondition:  &simulator.NilOutputCondition{},
			OutputFunction:   &simulator.NilOutputFunction{},
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
func RunSMC(d *data.SiteData, config SMCConfig) *SMCResult {
	nParams := len(config.Priors)

	model := analysis.SMCParticleModel{
		Build: func(N int, np int) *analysis.SMCInnerSimConfig {
			return buildInnerSimConfig(d, N, np)
		},
	}

	result := analysis.RunSMCInference(analysis.AppliedSMCInference{
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

	_ = nParams
	return result
}

// RunSMCSafe wraps RunSMC with panic recovery, returning nil and the
// error message if the inner simulation panics.
func RunSMCSafe(d *data.SiteData, config SMCConfig) (result *SMCResult, err error) {
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

// logSumExp is kept for backward compatibility in tests.
func logSumExp(x []float64) float64 {
	return stdinf.LogSumExp(x)
}

// computePosterior is kept for backward compatibility.
func computePosterior(
	paramNames []string,
	particleParams [][]float64,
	logLiks []float64,
	predictions [][]float64,
) *SMCResult {
	return stdinf.ComputePosterior(paramNames, particleParams, logLiks, predictions)
}

// choleskyDecomp is kept for backward compatibility in tests.
func choleskyDecomp(a []float64, d int) []float64 {
	// Inline since it's unexported in stochadex
	L := make([]float64, d*d)
	for i := range d {
		for j := 0; j <= i; j++ {
			sum := 0.0
			for k := 0; k < j; k++ {
				sum += L[i*d+k] * L[j*d+k]
			}
			if i == j {
				val := a[i*d+i] - sum
				if val <= 0 {
					return nil
				}
				L[i*d+j] = math.Sqrt(val)
			} else {
				L[i*d+j] = (a[i*d+j] - sum) / L[j*d+j]
			}
		}
	}
	return L
}
