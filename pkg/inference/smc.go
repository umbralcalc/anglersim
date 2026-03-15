package inference

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"

	"github.com/umbralcalc/anglersim/pkg/data"
	"github.com/umbralcalc/anglersim/pkg/population"
	"github.com/umbralcalc/stochadex/pkg/general"
	stdinf "github.com/umbralcalc/stochadex/pkg/inference"
	"github.com/umbralcalc/stochadex/pkg/simulator"
)

// SMCConfig holds configuration for the SMC inference run.
type SMCConfig struct {
	NumParticles int
	NumRounds    int
	Seed         uint64
	Priors       []Prior
	ParamNames   []string
	Verbose      bool
}

// SMCResult holds the posterior estimates from SMC inference.
type SMCResult struct {
	ParamNames     []string
	PosteriorMean  []float64
	PosteriorStd   []float64
	PosteriorCov   []float64   // d*d flattened row-major
	LogMarginalLik float64
	Predictions    [][]float64 // [T][N] predicted log-densities per particle
	ParticleParams [][]float64 // [N][d] final round particle parameters
	ParticleLogLik []float64   // [N] final round cumulative log-likelihoods
	Weights        []float64   // [N] normalised importance weights
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

// Prior type codes for params-based configuration.
const (
	PriorTypeUniform         = 0
	PriorTypeTruncatedNormal = 1
	PriorTypeHalfNormal      = 2
	PriorTypeLogNormal       = 3
)

// PriorParamsStride is the number of float64 values per prior in the
// prior_params encoding. Each prior type uses a subset:
//
//	Uniform (0):         [lo, hi, 0, 0]
//	TruncatedNormal (1): [mu, sigma, lo, hi]
//	HalfNormal (2):      [sigma, 0, 0, 0]
//	LogNormal (3):       [mu, sigma, 0, 0]
const PriorParamsStride = 4

// DecodePriors builds a []Prior from params-encoded type codes and
// parameter values. prior_types has length d, prior_params has length
// 4*d (PriorParamsStride per prior).
func DecodePriors(priorTypes, priorParams []float64) []Prior {
	d := len(priorTypes)
	priors := make([]Prior, d)
	for i := range d {
		pp := priorParams[i*PriorParamsStride : (i+1)*PriorParamsStride]
		switch int(priorTypes[i]) {
		case PriorTypeUniform:
			priors[i] = &UniformPrior{Lo: pp[0], Hi: pp[1]}
		case PriorTypeTruncatedNormal:
			priors[i] = &TruncatedNormalPrior{
				Mu: pp[0], Sigma: pp[1], Lo: pp[2], Hi: pp[3],
			}
		case PriorTypeHalfNormal:
			priors[i] = &HalfNormalPrior{Sigma: pp[0]}
		case PriorTypeLogNormal:
			priors[i] = &LogNormalPrior{Mu: pp[0], Sigma: pp[1]}
		default:
			panic(fmt.Sprintf("unknown prior type code: %v", priorTypes[i]))
		}
	}
	return priors
}

// EncodePriors converts a []Prior into params-compatible slices
// (prior_types and prior_params) for YAML configuration.
func EncodePriors(priors []Prior) (priorTypes, priorParams []float64) {
	d := len(priors)
	priorTypes = make([]float64, d)
	priorParams = make([]float64, d*PriorParamsStride)
	for i, p := range priors {
		pp := priorParams[i*PriorParamsStride : (i+1)*PriorParamsStride]
		switch v := p.(type) {
		case *UniformPrior:
			priorTypes[i] = PriorTypeUniform
			pp[0], pp[1] = v.Lo, v.Hi
		case *TruncatedNormalPrior:
			priorTypes[i] = PriorTypeTruncatedNormal
			pp[0], pp[1], pp[2], pp[3] = v.Mu, v.Sigma, v.Lo, v.Hi
		case *HalfNormalPrior:
			priorTypes[i] = PriorTypeHalfNormal
			pp[0] = v.Sigma
		case *LogNormalPrior:
			priorTypes[i] = PriorTypeLogNormal
			pp[0], pp[1] = v.Mu, v.Sigma
		default:
			panic(fmt.Sprintf("unknown prior type: %T", p))
		}
	}
	return
}

// NewNormalDataComparison returns a DataComparisonIteration with a
// NormalLikelihoodDistribution. This helper avoids import-alias conflicts
// when stochadex/pkg/inference and anglersim/pkg/inference are both needed
// (e.g. in the YAML code-generation path).
func NewNormalDataComparison() simulator.Iteration {
	return &stdinf.DataComparisonIteration{
		Likelihood: &stdinf.NormalLikelihoodDistribution{},
	}
}

// StateSliceIteration extracts a contiguous slice from an upstream
// partition's state, copying the values to avoid mutation of the
// received data. Use via params_from_upstream to receive the full
// upstream state as "latest_values", with "offset" and "width"
// specifying the slice bounds.
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

// ---------------------------------------------------------------------------
// SMC decomposed iterations
//
// The SMC inference is decomposed into three outer partitions:
//
//   [0] SMCProposalIteration            — generates N particle proposals
//   [1] EmbeddedSimulationRunIteration  — evaluates particles via inner sim
//   [2] SMCPosteriorIteration           — computes importance-weighted posterior
//
// The inner simulation (wrapped by the embedded iteration) contains:
//
//   [0]      observed_data  (FromStorageIteration)
//   [1]      covariates     (FromStorageIteration)
//   [2+2p]   ricker_p       (RickerIteration)
//   [2+2p+1] loglike_p      (DataComparisonIteration)
//
// Particle parameters are forwarded from the proposal partition to
// inner ricker/loglike partitions via indexed params_from_upstream.
// Channel dependencies: proposals → embedded_sim → posterior.
// Proposal feedback reads smc_posterior's state history (one-step lag).
// ---------------------------------------------------------------------------

// SMCProposalIteration generates N particle proposals at each step.
// On step 0 it draws from the prior. On subsequent steps it draws
// from a multivariate normal centred on the previous posterior,
// read from the smc_posterior partition's state history.
//
// State layout: [particle_params(N*d)] flattened row-major.
// State width: N*d.
//
// Params:
//
//	num_particles:      [N]
//	prior_types:        [type codes]
//	prior_params:       [4 values per prior]
//	posterior_partition: [partition_index] (via params_as_partitions)
//	verbose:            [0 or 1]
type SMCProposalIteration struct {
	Priors []Prior

	rng                  *rand.Rand
	numParticles         int
	nParams              int
	posteriorPartitionIdx int
	verbose              bool
}

func (s *SMCProposalIteration) Configure(
	partitionIndex int,
	settings *simulator.Settings,
) {
	iterParams := settings.Iterations[partitionIndex].Params
	seed := settings.Iterations[partitionIndex].Seed
	s.rng = rand.New(rand.NewPCG(seed, seed+1))
	s.verbose = iterParams.GetIndex("verbose", 0) > 0
	s.numParticles = int(iterParams.GetIndex("num_particles", 0))
	if s.numParticles == 0 {
		panic("SMCProposalIteration: num_particles must be set")
	}
	if s.Priors == nil {
		priorTypes, ok1 := iterParams.GetOk("prior_types")
		priorParams, ok2 := iterParams.GetOk("prior_params")
		if ok1 && ok2 {
			s.Priors = DecodePriors(priorTypes, priorParams)
		} else {
			panic("SMCProposalIteration: priors must be set")
		}
	}
	s.nParams = len(s.Priors)
	s.posteriorPartitionIdx = int(iterParams.GetIndex("posterior_partition", 0))
}

func (s *SMCProposalIteration) Iterate(
	params *simulator.Params,
	partitionIndex int,
	stateHistories []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) []float64 {
	round := timestepsHistory.CurrentStepNumber
	N := s.numParticles
	d := s.nParams

	particleParams := make([][]float64, N)
	if round == 1 {
		for p := range N {
			pp := make([]float64, d)
			for j, prior := range s.Priors {
				pp[j] = prior.Sample(s.rng)
			}
			particleParams[p] = pp
		}
	} else {
		// Read posterior from previous step's state history
		prevPosterior := stateHistories[s.posteriorPartitionIdx].Values.RawRowView(0)
		proposalMean := prevPosterior[:d]
		proposalCov := regulariseCov(
			prevPosterior[d:d+d*d], d, s.Priors,
		)
		particleParams = sampleMultivariateNormal(
			s.rng, N, proposalMean, proposalCov, s.Priors,
		)
	}

	if s.verbose {
		fmt.Printf("Round %d: drawing %d particles...\n", round, N)
	}

	// Pack flat: [p0_param0, p0_param1, ..., pN_paramD]
	state := make([]float64, N*d)
	for p := range N {
		copy(state[p*d:(p+1)*d], particleParams[p])
	}
	return state
}

// SMCPosteriorIteration computes importance-weighted posterior
// statistics from particle log-likelihoods and parameters received
// via params_from_upstream channels.
//
// State layout: [posterior_mean(d) | posterior_cov(d²) | log_marginal_lik(1)]
// State width: d + d² + 1.
//
// Params:
//
//	num_particles:    [N]
//	num_params:       [d]
//	particle_loglikes: [N values] (via params_from_upstream)
//	particle_params:   [N*d flat] (via params_from_upstream)
//	verbose:           [0 or 1]
type SMCPosteriorIteration struct {
	ParamNames []string

	numParticles int
	nParams      int
	verbose      bool
}

func (s *SMCPosteriorIteration) Configure(
	partitionIndex int,
	settings *simulator.Settings,
) {
	iterParams := settings.Iterations[partitionIndex].Params
	s.numParticles = int(iterParams.GetIndex("num_particles", 0))
	s.nParams = int(iterParams.GetIndex("num_params", 0))
	s.verbose = iterParams.GetIndex("verbose", 0) > 0
	if s.numParticles == 0 || s.nParams == 0 {
		panic("SMCPosteriorIteration: num_particles and num_params must be set")
	}
	if len(s.ParamNames) == 0 {
		s.ParamNames = make([]string, s.nParams)
		for i := range s.nParams {
			s.ParamNames[i] = fmt.Sprintf("param_%d", i)
		}
	}
}

func (s *SMCPosteriorIteration) Iterate(
	params *simulator.Params,
	partitionIndex int,
	stateHistories []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) []float64 {
	N := s.numParticles
	d := s.nParams

	logLiks := params.Get("particle_loglikes")
	proposalFlat := params.Get("particle_params")

	// Unflatten params
	particleParams := make([][]float64, N)
	for p := range N {
		particleParams[p] = make([]float64, d)
		copy(particleParams[p], proposalFlat[p*d:(p+1)*d])
	}

	result := computePosterior(s.ParamNames, particleParams, logLiks, nil)

	if s.verbose {
		fmt.Printf("  Log marginal likelihood: %.4f\n", result.LogMarginalLik)
		for i, name := range s.ParamNames {
			fmt.Printf("  %-22s mean=%.4f std=%.4f\n",
				name, result.PosteriorMean[i], result.PosteriorStd[i])
		}
	}

	// Pack state: [mean(d) | cov(d²) | log_marginal_lik(1)]
	state := make([]float64, d+d*d+1)
	copy(state[:d], result.PosteriorMean)
	copy(state[d:d+d*d], result.PosteriorCov)
	state[d+d*d] = result.LogMarginalLik
	return state
}

// PosteriorStateWidth returns the state width for SMCPosteriorIteration.
func PosteriorStateWidth(nParams int) int {
	return nParams + nParams*nParams + 1
}

// EmbeddedStateWidth returns the state width for the embedded inner
// simulation's concatenated output.
func EmbeddedStateWidth(numParticles, numCovariates int) int {
	return 1 + numCovariates + 2*numParticles
}

// innerRickerIdx returns the partition index of particle p's ricker
// iteration in the inner simulation.
func innerRickerIdx(p int) int { return 2 + 2*p }

// innerLoglikeIdx returns the partition index of particle p's loglike
// iteration in the inner simulation.
func innerLoglikeIdx(p int) int { return 2 + 2*p + 1 }

// loglikeOutputOffset returns the offset of particle p's loglike value
// in the embedded simulation's concatenated output state.
func loglikeOutputOffset(p, numCovariates int) int {
	return 1 + numCovariates + 2*p + 1
}

// buildInnerSimulation constructs the inner stochadex simulation
// settings and implementations for N parallel particle evaluations.
// The inner simulation is designed to be wrapped by
// EmbeddedSimulationRunIteration, which forwards particle parameters
// from the outer proposal partition via indexed params_from_upstream.
//
// Inner partition layout:
//
//	[0]      observed_data  (FromStorageIteration)
//	[1]      covariates     (FromStorageIteration)
//	[2+2p]   ricker_p       (RickerIteration)
//	[2+2p+1] loglike_p      (DataComparisonIteration)
func buildInnerSimulation(
	d *data.SiteData,
	N int,
) (*simulator.Settings, *simulator.Implementations) {
	T := len(d.Years)
	totalPartitions := 2 + 2*N

	iterSettings := make([]simulator.IterationSettings, totalPartitions)
	iterations := make([]simulator.Iteration, totalPartitions)

	// [0] observed_data
	iterSettings[0] = simulator.IterationSettings{
		Name:              "observed_data",
		Params:            simulator.NewParams(map[string][]float64{}),
		InitStateValues:   d.LogDensity[0],
		StateWidth:        1,
		StateHistoryDepth: 2,
		Seed:              0,
	}
	iterations[0] = &general.FromStorageIteration{Data: d.LogDensity}

	// [1] covariates
	iterSettings[1] = simulator.IterationSettings{
		Name:              "covariates",
		Params:            simulator.NewParams(map[string][]float64{}),
		InitStateValues:   d.Covariates[0],
		StateWidth:        d.NumCovariates,
		StateHistoryDepth: 2,
		Seed:              0,
	}
	iterations[1] = &general.FromStorageIteration{Data: d.Covariates}

	for p := range N {
		ri := innerRickerIdx(p)
		li := innerLoglikeIdx(p)

		// ricker_p — params forwarded from embedded sim each step
		iterSettings[ri] = simulator.IterationSettings{
			Name: fmt.Sprintf("ricker_%d", p),
			Params: simulator.NewParams(map[string][]float64{
				"growth_rate":            {0},
				"density_dependence":     {0},
				"covariate_coefficients": {0, 0, 0},
				"covariates":             d.Covariates[0],
				"process_noise_sd":       {0},
			}),
			ParamsFromUpstream: map[string]simulator.UpstreamConfig{
				"covariates": {Upstream: 1},
			},
			InitStateValues:   d.LogDensity[0],
			StateWidth:        1,
			StateHistoryDepth: 2,
			Seed:              uint64(p + 1),
		}
		iterations[ri] = &population.RickerIteration{}

		// loglike_p — variance forwarded from embedded sim each step
		iterSettings[li] = simulator.IterationSettings{
			Name: fmt.Sprintf("loglike_%d", p),
			Params: simulator.NewParams(map[string][]float64{
				"mean":               d.LogDensity[0],
				"variance":           {1.0}, // placeholder, forwarded
				"latest_data_values": d.LogDensity[0],
				"cumulative":         {1},
				"burn_in_steps":      {0},
			}),
			ParamsFromUpstream: map[string]simulator.UpstreamConfig{
				"mean":               {Upstream: ri},
				"latest_data_values": {Upstream: 0},
			},
			InitStateValues:   []float64{0.0},
			StateWidth:        1,
			StateHistoryDepth: 2,
			Seed:              0,
		}
		iterations[li] = &stdinf.DataComparisonIteration{
			Likelihood: &stdinf.NormalLikelihoodDistribution{},
		}
	}

	settings := &simulator.Settings{
		Iterations:            iterSettings,
		InitTimeValue:         d.Years[0],
		TimestepsHistoryDepth: 2,
	}
	settings.Init()

	implementations := &simulator.Implementations{
		Iterations:      iterations,
		OutputCondition: &simulator.NilOutputCondition{},
		OutputFunction:  &simulator.NilOutputFunction{},
		TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
			MaxNumberOfSteps: T - 1,
		},
		TimestepFunction: &general.FromStorageTimestepFunction{
			Data: d.Years,
		},
	}

	return settings, implementations
}

// buildEmbeddedParamsFromUpstream constructs the params_from_upstream
// map for the embedded simulation partition. Each particle's ricker
// and loglike parameters are routed from the proposal partition [0]
// using indexed upstream configs.
//
// Ricker params (per particle p, d params starting at p*d):
//
//	ricker_p/growth_rate            ← [p*d + 0]
//	ricker_p/density_dependence     ← [p*d + 1]
//	ricker_p/covariate_coefficients ← [p*d + 2, p*d + 3, p*d + 4]
//	ricker_p/process_noise_sd       ← [p*d + 5]
//
// Loglike params:
//
//	loglike_p/variance              ← [p*d + 6]
func buildEmbeddedParamsFromUpstream(
	N, nParams int,
) map[string]simulator.UpstreamConfig {
	upstream := make(map[string]simulator.UpstreamConfig)
	for p := range N {
		base := p * nParams
		upstream[fmt.Sprintf("ricker_%d/growth_rate", p)] = simulator.UpstreamConfig{
			Upstream: 0, Indices: []int{base + 0},
		}
		upstream[fmt.Sprintf("ricker_%d/density_dependence", p)] = simulator.UpstreamConfig{
			Upstream: 0, Indices: []int{base + 1},
		}
		upstream[fmt.Sprintf("ricker_%d/covariate_coefficients", p)] = simulator.UpstreamConfig{
			Upstream: 0, Indices: []int{base + 2, base + 3, base + 4},
		}
		upstream[fmt.Sprintf("ricker_%d/process_noise_sd", p)] = simulator.UpstreamConfig{
			Upstream: 0, Indices: []int{base + 5},
		}
		upstream[fmt.Sprintf("loglike_%d/variance", p)] = simulator.UpstreamConfig{
			Upstream: 0, Indices: []int{base + 6},
		}
	}
	return upstream
}

// RunSMC runs iterated importance sampling for the Ricker model as a
// stochadex simulation. The outer simulation has three partitions:
//
//	[0] smc_proposals  (SMCProposalIteration)
//	[1] smc_sim        (EmbeddedSimulationRunIteration wrapping inner sim)
//	[2] smc_posterior   (SMCPosteriorIteration)
//
// The inner simulation evaluates all N particles through the data.
// Particle parameters are forwarded from proposals to the inner sim
// via indexed params_from_upstream on the embedded partition.
//
// Channel deps: proposals → embedded_sim → posterior.
// Proposal feedback reads smc_posterior's state history (one-step lag).
func RunSMC(d *data.SiteData, config SMCConfig) *SMCResult {
	N := config.NumParticles
	nParams := len(config.Priors)
	numCov := d.NumCovariates
	posteriorWidth := PosteriorStateWidth(nParams)
	embeddedWidth := EmbeddedStateWidth(N, numCov)

	verboseFlag := 0.0
	if config.Verbose {
		verboseFlag = 1.0
	}
	priorTypes, priorParams := EncodePriors(config.Priors)

	// Init states
	proposalInit := make([]float64, N*nParams)
	embeddedInit := make([]float64, embeddedWidth)
	posteriorInit := make([]float64, posteriorWidth)
	for j := range nParams {
		posteriorInit[nParams+j*nParams+j] = 1.0
	}

	// Build inner simulation
	innerSettings, innerImpl := buildInnerSimulation(d, N)

	// Build indexed upstream wiring for embedded sim
	embeddedUpstream := buildEmbeddedParamsFromUpstream(N, nParams)

	// Build loglike extraction indices for posterior
	loglikeIndices := make([]int, N)
	for p := range N {
		loglikeIndices[p] = loglikeOutputOffset(p, numCov)
	}

	// Embedded sim params
	embeddedParams := map[string][]float64{
		"init_time_value": {d.Years[0]},
		"burn_in_steps":   {0},
	}

	iterSettings := []simulator.IterationSettings{
		// [0] smc_proposals
		{
			Name: "smc_proposals",
			Params: simulator.NewParams(map[string][]float64{
				"verbose":            {verboseFlag},
				"num_particles":      {float64(N)},
				"prior_types":        priorTypes,
				"prior_params":       priorParams,
				"posterior_partition": {2},
			}),
			InitStateValues:   proposalInit,
			StateWidth:        N * nParams,
			StateHistoryDepth: 2,
			Seed:              config.Seed,
		},
		// [1] smc_sim (EmbeddedSimulationRunIteration)
		{
			Name:               "smc_sim",
			Params:             simulator.NewParams(embeddedParams),
			ParamsFromUpstream: embeddedUpstream,
			InitStateValues:    embeddedInit,
			StateWidth:         embeddedWidth,
			StateHistoryDepth:  2,
			Seed:               config.Seed + 100,
		},
		// [2] smc_posterior
		{
			Name: "smc_posterior",
			Params: simulator.NewParams(map[string][]float64{
				"verbose":           {verboseFlag},
				"num_particles":     {float64(N)},
				"num_params":        {float64(nParams)},
				"particle_loglikes": make([]float64, N),
				"particle_params":   proposalInit,
			}),
			ParamsFromUpstream: map[string]simulator.UpstreamConfig{
				"particle_loglikes": {Upstream: 1, Indices: loglikeIndices},
				"particle_params":   {Upstream: 0},
			},
			InitStateValues:   posteriorInit,
			StateWidth:        posteriorWidth,
			StateHistoryDepth: 2,
			Seed:              0,
		},
	}

	iterations := []simulator.Iteration{
		&SMCProposalIteration{Priors: config.Priors},
		general.NewEmbeddedSimulationRunIteration(innerSettings, innerImpl),
		&SMCPosteriorIteration{ParamNames: config.ParamNames},
	}

	settings := &simulator.Settings{
		Iterations:            iterSettings,
		InitTimeValue:         0.0,
		TimestepsHistoryDepth: 2,
	}
	settings.Init()

	for i, iter := range iterations {
		iter.Configure(i, settings)
	}

	store := simulator.NewStateTimeStorage()
	implementations := &simulator.Implementations{
		Iterations:      iterations,
		OutputCondition: &simulator.EveryStepOutputCondition{},
		OutputFunction:  &simulator.StateTimeStorageOutputFunction{Store: store},
		TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
			MaxNumberOfSteps: config.NumRounds,
		},
		TimestepFunction: &simulator.ConstantTimestepFunction{Stepsize: 1.0},
	}

	coordinator := simulator.NewPartitionCoordinator(settings, implementations)
	coordinator.Run()

	// Extract final round results from stored output
	proposalVals := store.GetValues("smc_proposals")
	embeddedVals := store.GetValues("smc_sim")
	if len(proposalVals) == 0 || len(embeddedVals) == 0 {
		return nil
	}

	// Final round particle params
	finalProposal := proposalVals[len(proposalVals)-1]
	particleParams := make([][]float64, N)
	for p := range N {
		particleParams[p] = make([]float64, nParams)
		copy(particleParams[p], finalProposal[p*nParams:(p+1)*nParams])
	}

	// Final round log-likelihoods
	finalEmbedded := embeddedVals[len(embeddedVals)-1]
	logLiks := make([]float64, N)
	for p := range N {
		ll := finalEmbedded[loglikeOutputOffset(p, numCov)]
		if math.IsNaN(ll) {
			ll = math.Inf(-1)
		}
		logLiks[p] = ll
	}

	result := computePosterior(config.ParamNames, particleParams, logLiks, nil)
	result.ParticleParams = particleParams
	result.ParticleLogLik = logLiks
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
	return result, nil
}

// computePosterior computes posterior statistics from weighted particles.
func computePosterior(
	paramNames []string,
	particleParams [][]float64,
	logLiks []float64,
	predictions [][]float64,
) *SMCResult {
	N := len(particleParams)
	nParams := len(particleParams[0])

	logMarginalLik := logSumExp(logLiks) - math.Log(float64(N))

	logWeights := make([]float64, N)
	copy(logWeights, logLiks)
	logZ := logSumExp(logWeights)
	weights := make([]float64, N)
	for i := range N {
		weights[i] = math.Exp(logWeights[i] - logZ)
	}

	postMean := make([]float64, nParams)
	for p := range N {
		for j := range nParams {
			postMean[j] += weights[p] * particleParams[p][j]
		}
	}

	postCov := make([]float64, nParams*nParams)
	for p := range N {
		for i := range nParams {
			di := particleParams[p][i] - postMean[i]
			for j := range nParams {
				dj := particleParams[p][j] - postMean[j]
				postCov[i*nParams+j] += weights[p] * di * dj
			}
		}
	}

	postStd := make([]float64, nParams)
	for j := range nParams {
		postStd[j] = math.Sqrt(postCov[j*nParams+j])
	}

	return &SMCResult{
		ParamNames:     paramNames,
		PosteriorMean:  postMean,
		PosteriorStd:   postStd,
		PosteriorCov:   postCov,
		LogMarginalLik: logMarginalLik,
		Predictions:    predictions,
		Weights:        weights,
	}
}

// sampleMultivariateNormal draws N samples from a multivariate normal,
// rejecting any samples that fall outside prior support.
func sampleMultivariateNormal(
	rng *rand.Rand,
	n int,
	mean []float64,
	covFlat []float64,
	priors []Prior,
) [][]float64 {
	d := len(mean)

	L := choleskyDecomp(covFlat, d)
	if L == nil {
		L = make([]float64, d*d)
		for i := range d {
			v := covFlat[i*d+i]
			if v <= 0 {
				v = 1.0
			}
			L[i*d+i] = math.Sqrt(v)
		}
	}

	samples := make([][]float64, n)
	for i := range n {
		for {
			z := make([]float64, d)
			for j := range d {
				z[j] = rng.NormFloat64()
			}
			x := make([]float64, d)
			for row := range d {
				x[row] = mean[row]
				for col := 0; col <= row; col++ {
					x[row] += L[row*d+col] * z[col]
				}
			}
			valid := true
			for j, p := range priors {
				if !p.InSupport(x[j]) {
					valid = false
					break
				}
			}
			if valid {
				samples[i] = x
				break
			}
		}
	}
	return samples
}

// choleskyDecomp computes the lower Cholesky factor of a d×d symmetric
// positive-definite matrix (row-major flat). Returns nil if not PD.
func choleskyDecomp(a []float64, d int) []float64 {
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

// regulariseCov adds a minimum diagonal to prevent posterior covariance
// collapse between importance sampling rounds.
func regulariseCov(cov []float64, d int, priors []Prior) []float64 {
	reg := make([]float64, len(cov))
	copy(reg, cov)
	for i := range d {
		var priorVar float64
		switch p := priors[i].(type) {
		case *UniformPrior:
			span := p.Hi - p.Lo
			priorVar = span * span / 12.0
		case *TruncatedNormalPrior:
			priorVar = p.Sigma * p.Sigma
		case *HalfNormalPrior:
			priorVar = p.Sigma * p.Sigma
		case *LogNormalPrior:
			priorVar = p.Sigma * p.Sigma
		default:
			priorVar = 1.0
		}
		floor := 0.01 * priorVar
		if reg[i*d+i] < floor {
			reg[i*d+i] = floor
		}
	}
	return reg
}

// logSumExp computes log(sum(exp(x))) with numerical stability.
func logSumExp(x []float64) float64 {
	if len(x) == 0 {
		return math.Inf(-1)
	}
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	if math.IsInf(maxVal, -1) {
		return math.Inf(-1)
	}
	sum := 0.0
	for _, v := range x {
		sum += math.Exp(v - maxVal)
	}
	return maxVal + math.Log(sum)
}

// WeightedQuantiles computes weighted quantiles from particle values.
func WeightedQuantiles(values, weights, probs []float64) []float64 {
	n := len(values)
	indices := make([]int, n)
	for i := range n {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		return values[indices[i]] < values[indices[j]]
	})

	cumWeight := make([]float64, n)
	cumWeight[0] = weights[indices[0]]
	for i := 1; i < n; i++ {
		cumWeight[i] = cumWeight[i-1] + weights[indices[i]]
	}
	total := cumWeight[n-1]
	for i := range n {
		cumWeight[i] /= total
	}

	result := make([]float64, len(probs))
	for q, p := range probs {
		for i := range n {
			if cumWeight[i] >= p {
				result[q] = values[indices[i]]
				break
			}
		}
	}
	return result
}
