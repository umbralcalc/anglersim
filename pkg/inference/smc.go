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
			"process_noise_sd", "obs_noise_sd",
		},
	}
}

// SMCRoundIteration implements simulator.Iteration where each step
// corresponds to one SMC importance sampling round. At each step it:
//  1. Generates particle proposals (prior on step 0, MVN from previous
//     posterior on subsequent steps)
//  2. Builds and runs an embedded inner simulation evaluating all
//     particles through the data
//  3. Computes the posterior and returns it as state
//
// State layout: [posterior_mean(d) | posterior_cov(d²) | log_marginal_lik(1)]
// State width: d + d² + 1
type SMCRoundIteration struct {
	SiteData     *data.SiteData
	Priors       []Prior
	ParamNames   []string
	NumParticles int

	rng        *rand.Rand
	nParams    int
	verbose    bool
	lastResult *SMCResult // detailed result from the most recent round
}

func (s *SMCRoundIteration) Configure(
	partitionIndex int,
	settings *simulator.Settings,
) {
	s.nParams = len(s.Priors)
	seed := settings.Iterations[partitionIndex].Seed
	s.rng = rand.New(rand.NewPCG(seed, seed+1))
	s.verbose = settings.Iterations[partitionIndex].Params.GetIndex(
		"verbose", 0) > 0
}

func (s *SMCRoundIteration) Iterate(
	params *simulator.Params,
	partitionIndex int,
	stateHistories []*simulator.StateHistory,
	timestepsHistory *simulator.CumulativeTimestepsHistory,
) []float64 {
	round := timestepsHistory.CurrentStepNumber
	d := s.nParams

	// Generate particle proposals
	particleParams := make([][]float64, s.NumParticles)
	if round == 0 {
		// First round: draw from prior
		for p := range s.NumParticles {
			pp := make([]float64, d)
			for j, prior := range s.Priors {
				pp[j] = prior.Sample(s.rng)
			}
			particleParams[p] = pp
		}
	} else {
		// Subsequent rounds: draw from MVN using previous posterior
		prevState := stateHistories[partitionIndex].Values.RawRowView(0)
		proposalMean := prevState[:d]
		proposalCov := regulariseCov(prevState[d:d+d*d], d, s.Priors)
		particleParams = sampleMultivariateNormal(
			s.rng, s.NumParticles, proposalMean, proposalCov, s.Priors,
		)
	}

	if s.verbose {
		fmt.Printf("Round %d: running %d particles...\n",
			round+1, s.NumParticles)
	}

	// Build and run the inner (embedded) simulation
	settings, implementations, store := buildSMCSimulation(
		s.SiteData, particleParams,
	)
	func() {
		defer func() { recover() }()
		coordinator := simulator.NewPartitionCoordinator(
			settings, implementations,
		)
		coordinator.Run()
	}()

	// Extract results from the inner simulation
	logLiks := make([]float64, s.NumParticles)
	for p := range s.NumParticles {
		name := fmt.Sprintf("loglike_%d", p)
		vals := store.GetValues(name)
		if len(vals) == 0 {
			logLiks[p] = math.Inf(-1)
		} else {
			logLiks[p] = vals[len(vals)-1][0]
			if math.IsNaN(logLiks[p]) {
				logLiks[p] = math.Inf(-1)
			}
		}
	}

	T := len(s.SiteData.Years)
	predictions := make([][]float64, T)
	for t := range T {
		predictions[t] = make([]float64, s.NumParticles)
	}
	for p := range s.NumParticles {
		name := fmt.Sprintf("ricker_%d", p)
		vals := store.GetValues(name)
		predictions[0][p] = s.SiteData.LogDensity[0][0]
		for t := range len(vals) {
			if t+1 < T {
				predictions[t+1][p] = vals[t][0]
			}
		}
	}

	// Compute posterior
	result := computePosterior(
		s.ParamNames, particleParams, logLiks, predictions,
	)
	result.ParticleParams = particleParams
	result.ParticleLogLik = logLiks
	s.lastResult = result

	if s.verbose {
		fmt.Printf("  Log marginal likelihood: %.4f\n", result.LogMarginalLik)
		for i, name := range s.ParamNames {
			fmt.Printf("  %-22s mean=%.4f std=%.4f\n",
				name, result.PosteriorMean[i], result.PosteriorStd[i])
		}
	}

	// Pack state: [posterior_mean | posterior_cov | log_marginal_lik]
	state := make([]float64, d+d*d+1)
	copy(state[:d], result.PosteriorMean)
	copy(state[d:d+d*d], result.PosteriorCov)
	state[d+d*d] = result.LogMarginalLik
	return state
}

// LastResult returns the detailed SMCResult from the most recent round.
func (s *SMCRoundIteration) LastResult() *SMCResult {
	return s.lastResult
}

// RunSMC runs iterated importance sampling for the Ricker model as a
// stochadex simulation. Each simulation step corresponds to one SMC round:
// drawing parameter proposals, running the full particle evaluation via
// an embedded inner simulation, and computing the posterior.
func RunSMC(d *data.SiteData, config SMCConfig) *SMCResult {
	nParams := len(config.Priors)
	stateWidth := nParams + nParams*nParams + 1

	// Initial state: zero mean, identity cov, zero log-marginal-lik
	initState := make([]float64, stateWidth)
	for j := range nParams {
		initState[nParams+j*nParams+j] = 1.0
	}

	smcIter := &SMCRoundIteration{
		SiteData:     d,
		Priors:       config.Priors,
		ParamNames:   config.ParamNames,
		NumParticles: config.NumParticles,
	}

	verboseFlag := 0.0
	if config.Verbose {
		verboseFlag = 1.0
	}

	settings := &simulator.Settings{
		Iterations: []simulator.IterationSettings{
			{
				Name: "smc_round",
				Params: simulator.NewParams(map[string][]float64{
					"verbose": {verboseFlag},
				}),
				InitStateValues:   initState,
				StateWidth:        stateWidth,
				StateHistoryDepth: 2,
				Seed:              config.Seed,
			},
		},
		InitTimeValue:         0.0,
		TimestepsHistoryDepth: 2,
	}
	settings.Init()

	smcIter.Configure(0, settings)

	implementations := &simulator.Implementations{
		Iterations:      []simulator.Iteration{smcIter},
		OutputCondition: &simulator.EveryStepOutputCondition{},
		OutputFunction:  &simulator.NilOutputFunction{},
		TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
			MaxNumberOfSteps: config.NumRounds,
		},
		TimestepFunction: &simulator.ConstantTimestepFunction{Stepsize: 1.0},
	}

	func() {
		defer func() { recover() }()
		coordinator := simulator.NewPartitionCoordinator(
			settings, implementations,
		)
		coordinator.Run()
	}()

	return smcIter.LastResult()
}

// Partition index helpers for the inner simulation layout:
//
//	[0]            observed_data  (FromStorageIteration)
//	[1]            covariates     (FromStorageIteration)
//	[2+3p]         params_p       (ParamValuesIteration)
//	[2+3p+1]       ricker_p       (RickerIteration)
//	[2+3p+2]       loglike_p      (DataComparisonIteration)
//	[2+3N]         log_norm       (PosteriorLogNormalisationIteration)
//	[2+3N+1]       posterior_mean (PosteriorMeanIteration)
//	[2+3N+2]       posterior_cov  (PosteriorCovarianceIteration)
func paramsIdx(p int) int  { return 2 + 3*p }
func rickerIdx(p int) int  { return 2 + 3*p + 1 }
func loglikeIdx(p int) int { return 2 + 3*p + 2 }
func logNormIdx(n int) int { return 2 + 3*n }
func postMeanIdx(n int) int { return 2 + 3*n + 1 }
func postCovIdx(n int) int  { return 2 + 3*n + 2 }

// buildSMCSimulation constructs the inner stochadex simulation with N
// parallel particle evaluation pipelines plus posterior aggregation
// partitions.
func buildSMCSimulation(
	d *data.SiteData,
	particleParams [][]float64,
) (*simulator.Settings, *simulator.Implementations, *simulator.StateTimeStorage) {
	N := len(particleParams)
	nParams := len(particleParams[0])
	T := len(d.Years)
	totalPartitions := 3*N + 5 // 2 data + 3N particles + 3 posterior

	iterSettings := make([]simulator.IterationSettings, totalPartitions)
	iterations := make([]simulator.Iteration, totalPartitions)

	// [0] observed_data — streams log-density observations
	iterSettings[0] = simulator.IterationSettings{
		Name:              "observed_data",
		Params:            simulator.NewParams(map[string][]float64{}),
		InitStateValues:   d.LogDensity[0],
		StateWidth:        1,
		StateHistoryDepth: 2,
		Seed:              0,
	}
	iterations[0] = &general.FromStorageIteration{Data: d.LogDensity}

	// [1] covariates — streams environmental covariates
	iterSettings[1] = simulator.IterationSettings{
		Name:              "covariates",
		Params:            simulator.NewParams(map[string][]float64{}),
		InitStateValues:   d.Covariates[0],
		StateWidth:        d.NumCovariates,
		StateHistoryDepth: 2,
		Seed:              0,
	}
	iterations[1] = &general.FromStorageIteration{Data: d.Covariates}

	// Per-particle partitions
	paramPartitionIndices := make([]float64, N)
	loglikePartitionIndices := make([]float64, N)

	for p := range N {
		pp := particleParams[p]
		pi := paramsIdx(p)
		ri := rickerIdx(p)
		li := loglikeIdx(p)

		paramPartitionIndices[p] = float64(pi)
		loglikePartitionIndices[p] = float64(li)

		// params_p — holds the 7 parameter values as state
		iterSettings[pi] = simulator.IterationSettings{
			Name: fmt.Sprintf("params_%d", p),
			Params: simulator.NewParams(map[string][]float64{
				"param_values": pp,
			}),
			InitStateValues:   pp,
			StateWidth:        nParams,
			StateHistoryDepth: 2,
			Seed:              0,
		}
		iterations[pi] = &general.ParamValuesIteration{}

		// ricker_p — population dynamics with this particle's parameters
		iterSettings[ri] = simulator.IterationSettings{
			Name: fmt.Sprintf("ricker_%d", p),
			Params: simulator.NewParams(map[string][]float64{
				"growth_rate":            {pp[0]},
				"density_dependence":     {pp[1]},
				"covariate_coefficients": {pp[2], pp[3], pp[4]},
				"covariates":             d.Covariates[0],
				"process_noise_sd":       {pp[5]},
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

		// loglike_p — cumulative log-likelihood comparing Ricker to observed
		obsSD := pp[6]
		iterSettings[li] = simulator.IterationSettings{
			Name: fmt.Sprintf("loglike_%d", p),
			Params: simulator.NewParams(map[string][]float64{
				"mean":               d.LogDensity[0],
				"variance":           {obsSD * obsSD},
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

	// Posterior aggregation partitions
	lnIdx := logNormIdx(N)
	pmIdx := postMeanIdx(N)
	pcIdx := postCovIdx(N)

	// [2+3N] log_norm — normalisation across particle log-likelihoods
	iterSettings[lnIdx] = simulator.IterationSettings{
		Name: "log_norm",
		Params: simulator.NewParams(map[string][]float64{
			"loglike_partitions":      loglikePartitionIndices,
			"past_discounting_factor": {0.001}, // effectively use only current step
		}),
		InitStateValues:   []float64{0.0},
		StateWidth:        1,
		StateHistoryDepth: 2,
		Seed:              0,
	}
	iterations[lnIdx] = &stdinf.PosteriorLogNormalisationIteration{}

	// [2+3N+1] posterior_mean — importance-weighted mean of parameters
	initMean := make([]float64, nParams)
	for p := range N {
		for j := range nParams {
			initMean[j] += particleParams[p][j]
		}
	}
	for j := range nParams {
		initMean[j] /= float64(N)
	}

	iterSettings[pmIdx] = simulator.IterationSettings{
		Name: "posterior_mean",
		Params: simulator.NewParams(map[string][]float64{
			"param_partitions":   paramPartitionIndices,
			"loglike_partitions": loglikePartitionIndices,
		}),
		ParamsFromUpstream: map[string]simulator.UpstreamConfig{
			"posterior_log_normalisation": {Upstream: lnIdx},
		},
		InitStateValues:   initMean,
		StateWidth:        nParams,
		StateHistoryDepth: 2,
		Seed:              0,
	}
	iterations[pmIdx] = &stdinf.PosteriorMeanIteration{
		Transform: stdinf.MeanTransform,
	}

	// [2+3N+2] posterior_cov — importance-weighted covariance
	initCov := make([]float64, nParams*nParams)
	for j := range nParams {
		initCov[j*nParams+j] = 1.0 // identity matrix
	}

	iterSettings[pcIdx] = simulator.IterationSettings{
		Name: "posterior_cov",
		Params: simulator.NewParams(map[string][]float64{
			"param_partitions":   paramPartitionIndices,
			"loglike_partitions": loglikePartitionIndices,
			"mean":               initMean,
		}),
		ParamsFromUpstream: map[string]simulator.UpstreamConfig{
			"posterior_log_normalisation": {Upstream: lnIdx},
			"mean":                        {Upstream: pmIdx},
		},
		InitStateValues:   initCov,
		StateWidth:        nParams * nParams,
		StateHistoryDepth: 2,
		Seed:              0,
	}
	iterations[pcIdx] = &stdinf.PosteriorCovarianceIteration{}

	// Build Settings
	settings := &simulator.Settings{
		Iterations:            iterSettings,
		InitTimeValue:         d.Years[0],
		TimestepsHistoryDepth: 2,
	}
	settings.Init()

	// Configure all iterations
	for i, iter := range iterations {
		iter.Configure(i, settings)
	}

	store := simulator.NewStateTimeStorage()
	implementations := &simulator.Implementations{
		Iterations:      iterations,
		OutputCondition: &simulator.EveryStepOutputCondition{},
		OutputFunction:  &simulator.StateTimeStorageOutputFunction{Store: store},
		TerminationCondition: &simulator.NumberOfStepsTerminationCondition{
			MaxNumberOfSteps: T - 1,
		},
		TimestepFunction: &general.FromStorageTimestepFunction{
			Data: d.Years,
		},
	}

	return settings, implementations, store
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

	// Log marginal likelihood estimate: log(mean(exp(logLiks)))
	logMarginalLik := logSumExp(logLiks) - math.Log(float64(N))

	// Normalised weights
	logWeights := make([]float64, N)
	copy(logWeights, logLiks)
	logZ := logSumExp(logWeights)
	weights := make([]float64, N)
	for i := range N {
		weights[i] = math.Exp(logWeights[i] - logZ)
	}

	// Posterior mean
	postMean := make([]float64, nParams)
	for p := range N {
		for j := range nParams {
			postMean[j] += weights[p] * particleParams[p][j]
		}
	}

	// Posterior covariance
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

	// Posterior std dev
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

	// Cholesky decomposition of covariance
	L := choleskyDecomp(covFlat, d)
	if L == nil {
		// Fallback: use diagonal with inflated variance
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
			// Draw standard normal
			z := make([]float64, d)
			for j := range d {
				z[j] = rng.NormFloat64()
			}
			// Transform: x = mean + L * z
			x := make([]float64, d)
			for row := range d {
				x[row] = mean[row]
				for col := 0; col <= row; col++ {
					x[row] += L[row*d+col] * z[col]
				}
			}
			// Check all priors
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
// collapse between importance sampling rounds. Uses 1% of prior variance
// as a floor for each diagonal element.
func regulariseCov(cov []float64, d int, priors []Prior) []float64 {
	reg := make([]float64, len(cov))
	copy(reg, cov)
	for i := range d {
		// Compute a reasonable floor from the prior's scale
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

// LogSumExpPublic computes log(sum(exp(x))) with numerical stability.
func LogSumExpPublic(x []float64) float64 {
	return logSumExp(x)
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
// Returns quantiles at the given probability levels (e.g., 0.025, 0.5, 0.975).
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
	// Normalise
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
