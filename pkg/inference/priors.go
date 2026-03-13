package inference

import (
	"math"
	"math/rand/v2"
)

// Prior defines a 1D prior distribution for a model parameter.
type Prior interface {
	Sample(rng *rand.Rand) float64
	LogPDF(x float64) float64
	InSupport(x float64) bool
}

// UniformPrior is a uniform distribution on [Lo, Hi].
type UniformPrior struct {
	Lo, Hi float64
}

func (p *UniformPrior) Sample(rng *rand.Rand) float64 {
	return p.Lo + rng.Float64()*(p.Hi-p.Lo)
}

func (p *UniformPrior) LogPDF(x float64) float64 {
	if x < p.Lo || x > p.Hi {
		return math.Inf(-1)
	}
	return -math.Log(p.Hi - p.Lo)
}

func (p *UniformPrior) InSupport(x float64) bool {
	return x >= p.Lo && x <= p.Hi
}

// TruncatedNormalPrior is a normal distribution truncated to [Lo, Hi].
type TruncatedNormalPrior struct {
	Mu, Sigma float64
	Lo, Hi    float64
}

func (p *TruncatedNormalPrior) Sample(rng *rand.Rand) float64 {
	for {
		x := rng.NormFloat64()*p.Sigma + p.Mu
		if x >= p.Lo && x <= p.Hi {
			return x
		}
	}
}

func (p *TruncatedNormalPrior) LogPDF(x float64) float64 {
	if x < p.Lo || x > p.Hi {
		return math.Inf(-1)
	}
	d := (x - p.Mu) / p.Sigma
	// Unnormalised (ignoring truncation constant, which is the same for all x in support)
	return -0.5*d*d - math.Log(p.Sigma) - 0.5*math.Log(2*math.Pi)
}

func (p *TruncatedNormalPrior) InSupport(x float64) bool {
	return x >= p.Lo && x <= p.Hi
}

// HalfNormalPrior is a half-normal distribution (x >= 0) with scale sigma.
type HalfNormalPrior struct {
	Sigma float64
}

func (p *HalfNormalPrior) Sample(rng *rand.Rand) float64 {
	return math.Abs(rng.NormFloat64()) * p.Sigma
}

func (p *HalfNormalPrior) LogPDF(x float64) float64 {
	if x < 0 {
		return math.Inf(-1)
	}
	d := x / p.Sigma
	return math.Log(2) - 0.5*d*d - math.Log(p.Sigma) - 0.5*math.Log(2*math.Pi)
}

func (p *HalfNormalPrior) InSupport(x float64) bool {
	return x >= 0
}

// LogNormalPrior is a log-normal distribution: log(x) ~ N(mu, sigma^2).
type LogNormalPrior struct {
	Mu, Sigma float64
}

func (p *LogNormalPrior) Sample(rng *rand.Rand) float64 {
	return math.Exp(rng.NormFloat64()*p.Sigma + p.Mu)
}

func (p *LogNormalPrior) LogPDF(x float64) float64 {
	if x <= 0 {
		return math.Inf(-1)
	}
	logX := math.Log(x)
	d := (logX - p.Mu) / p.Sigma
	return -0.5*d*d - logX - math.Log(p.Sigma) - 0.5*math.Log(2*math.Pi)
}

func (p *LogNormalPrior) InSupport(x float64) bool {
	return x > 0
}

// DefaultRickerPriors returns the default priors for the 7 Ricker model parameters:
// [growth_rate, density_dependence, beta_flow, beta_temp, beta_do, process_noise_sd, obs_noise_sd]
func DefaultRickerPriors() []Prior {
	return []Prior{
		&TruncatedNormalPrior{Mu: 0.5, Sigma: 1.0, Lo: -2.0, Hi: 5.0}, // growth_rate
		&LogNormalPrior{Mu: 0.5, Sigma: 1.5},                           // density_dependence
		&TruncatedNormalPrior{Mu: 0, Sigma: 0.3, Lo: -2.0, Hi: 2.0},   // beta_flow
		&TruncatedNormalPrior{Mu: 0, Sigma: 0.3, Lo: -2.0, Hi: 2.0},   // beta_temp
		&TruncatedNormalPrior{Mu: 0, Sigma: 0.3, Lo: -2.0, Hi: 2.0},   // beta_do
		&HalfNormalPrior{Sigma: 0.5},                                    // process_noise_sd
		&HalfNormalPrior{Sigma: 0.5},                                    // obs_noise_sd
	}
}
