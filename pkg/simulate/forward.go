package simulate

import (
	"math"
	"math/rand/v2"
	"sort"

	"github.com/umbralcalc/anglersim/pkg/data"
)

// SiteFittedParams holds the posterior estimates for a single site.
type SiteFittedParams struct {
	SiteID int
	Mean   []float64 // [7] posterior means
	Std    []float64 // [7] posterior stds
}

// ProjectionConfig controls the forward simulation.
type ProjectionConfig struct {
	Horizon             int     // projection years (default 20)
	NumSims             int     // Monte Carlo trajectories per site (default 500)
	Seed                uint64
	ExtinctionThreshold float64 // log-density threshold for functional extinction
}

// DefaultProjectionConfig returns sensible defaults.
func DefaultProjectionConfig() ProjectionConfig {
	return ProjectionConfig{
		Horizon:             20,
		NumSims:             500,
		Seed:                42,
		ExtinctionThreshold: -6.9, // ~0.001 fish/m²
	}
}

// SiteProjection holds the projection results for one site under one scenario.
type SiteProjection struct {
	SiteID       int
	Area         string
	ScenarioName string

	// Historical baseline
	BaselineMeanLogN float64 // mean log-density over last 5 observed years

	// Projection summary per year
	ProjYears []float64
	MeanLogN  []float64 // [horizon]
	MedianLogN []float64
	Lo90      []float64
	Hi90      []float64
	Lo50      []float64
	Hi50      []float64

	// Derived metrics at horizon
	MeanDensityChangePct float64 // mean % change in density at horizon vs baseline
	ExtinctionProb       float64 // fraction of sims below threshold at horizon
	RecoveryProb         float64 // fraction of sims above baseline mean at horizon
}

// ProjectSite runs Monte Carlo forward simulation for a single site
// under a given scenario.
func ProjectSite(
	sd *data.SiteData,
	params *SiteFittedParams,
	scenario Scenario,
	config ProjectionConfig,
) *SiteProjection {
	T := len(sd.Years)
	if T < 2 {
		return nil
	}

	proj := &SiteProjection{
		SiteID:       sd.SiteID,
		Area:         sd.Area,
		ScenarioName: scenario.Name,
	}

	// Compute baseline: mean log-density over last 5 years (or all if < 5)
	nBaseline := 5
	if T < nBaseline {
		nBaseline = T
	}
	baselineSum := 0.0
	for i := T - nBaseline; i < T; i++ {
		baselineSum += sd.LogDensity[i][0]
	}
	proj.BaselineMeanLogN = baselineSum / float64(nBaseline)

	// Compute historical covariate means for percentage-based modifiers
	covMeans := make([]float64, sd.NumCovariates)
	for t := range T {
		for k := range sd.NumCovariates {
			covMeans[k] += sd.Covariates[t][k]
		}
	}
	for k := range sd.NumCovariates {
		covMeans[k] /= float64(T)
	}

	// Compute additive covariate shifts from scenario modifier
	// Covariate order: [flow, temp, DO]
	covShifts := make([]float64, sd.NumCovariates)
	if sd.NumCovariates >= 1 {
		covShifts[0] = scenario.Modifier.FlowPct * covMeans[0] // flow: percentage
	}
	if sd.NumCovariates >= 2 {
		covShifts[1] = scenario.Modifier.TempShift // temp: absolute
	}
	if sd.NumCovariates >= 3 {
		covShifts[2] = scenario.Modifier.DOPct * covMeans[2] // DO: percentage
	}

	lastLogN := sd.LogDensity[T-1][0]
	lastYear := sd.Years[T-1]
	horizon := config.Horizon
	nSims := config.NumSims

	rng := rand.New(rand.NewPCG(config.Seed+uint64(sd.SiteID), 0))

	// Prior support bounds for parameter sampling
	type bound struct{ lo, hi float64 }
	priorBounds := []bound{
		{-2.0, 5.0},      // growth_rate
		{0.0, math.Inf(1)}, // density_dependence (LogNormal, positive)
		{-2.0, 2.0},      // beta_flow
		{-2.0, 2.0},      // beta_temp
		{-2.0, 2.0},      // beta_do
		{0.0, math.Inf(1)}, // process_noise_sd (HalfNormal, positive)
		{0.0, math.Inf(1)}, // obs_noise_var (LogNormal, positive)
	}

	// Run trajectories
	trajectories := make([][]float64, nSims)
	for s := range nSims {
		// Sample parameters from posterior (independent normals, reject outside support)
		pp := make([]float64, 7)
		for j := range 7 {
			for {
				pp[j] = params.Mean[j] + rng.NormFloat64()*params.Std[j]
				if pp[j] >= priorBounds[j].lo && pp[j] <= priorBounds[j].hi {
					break
				}
			}
		}

		r0 := pp[0]
		alpha := pp[1]
		betas := pp[2:5]
		sigma := pp[5]

		traj := make([]float64, horizon)
		logN := lastLogN
		for t := range horizon {
			// Bootstrap resample covariates from historical distribution
			histIdx := rng.IntN(T)
			covs := make([]float64, sd.NumCovariates)
			copy(covs, sd.Covariates[histIdx])

			// Apply scenario shifts
			for k := range sd.NumCovariates {
				covs[k] += covShifts[k]
			}

			// Ricker dynamics
			envEffect := 0.0
			for k := 0; k < 3 && k < len(covs); k++ {
				envEffect += betas[k] * covs[k]
			}
			logN = logN + r0 + envEffect - alpha*math.Exp(logN) + rng.NormFloat64()*sigma

			// Clip to prevent divergence
			if logN > 20 {
				logN = 20
			} else if logN < -20 {
				logN = -20
			}

			traj[t] = logN
		}
		trajectories[s] = traj
	}

	// Compute summary statistics per projection year
	proj.ProjYears = make([]float64, horizon)
	proj.MeanLogN = make([]float64, horizon)
	proj.MedianLogN = make([]float64, horizon)
	proj.Lo90 = make([]float64, horizon)
	proj.Hi90 = make([]float64, horizon)
	proj.Lo50 = make([]float64, horizon)
	proj.Hi50 = make([]float64, horizon)

	for t := range horizon {
		proj.ProjYears[t] = lastYear + float64(t+1)

		vals := make([]float64, nSims)
		sum := 0.0
		for s := range nSims {
			vals[s] = trajectories[s][t]
			sum += vals[s]
		}
		sort.Float64s(vals)

		proj.MeanLogN[t] = sum / float64(nSims)
		proj.MedianLogN[t] = vals[nSims/2]
		proj.Lo90[t] = vals[int(float64(nSims)*0.05)]
		proj.Hi90[t] = vals[int(float64(nSims)*0.95)]
		proj.Lo50[t] = vals[nSims/4]
		proj.Hi50[t] = vals[3*nSims/4]
	}

	// Derived metrics at final projection year
	finalVals := make([]float64, nSims)
	for s := range nSims {
		finalVals[s] = trajectories[s][horizon-1]
	}

	// Mean density change (in percentage, computed in density space)
	baselineDensity := math.Exp(proj.BaselineMeanLogN)
	meanFinalLogN := proj.MeanLogN[horizon-1]
	projectedDensity := math.Exp(meanFinalLogN)
	if baselineDensity > 0 {
		proj.MeanDensityChangePct = 100 * (projectedDensity - baselineDensity) / baselineDensity
	}

	// Extinction probability
	extinct := 0
	recovered := 0
	for _, v := range finalVals {
		if v < config.ExtinctionThreshold {
			extinct++
		}
		if v > proj.BaselineMeanLogN {
			recovered++
		}
	}
	proj.ExtinctionProb = float64(extinct) / float64(nSims)
	proj.RecoveryProb = float64(recovered) / float64(nSims)

	return proj
}
