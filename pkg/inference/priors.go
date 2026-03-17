package inference

import (
	stdinf "github.com/umbralcalc/stochadex/pkg/inference"
)

// Type aliases re-exporting prior types from stochadex.
type Prior = stdinf.Prior
type UniformPrior = stdinf.UniformPrior
type TruncatedNormalPrior = stdinf.TruncatedNormalPrior
type HalfNormalPrior = stdinf.HalfNormalPrior
type LogNormalPrior = stdinf.LogNormalPrior

// DefaultRickerPriors returns the default priors for the 7 Ricker model parameters:
// [growth_rate, density_dependence, beta_flow, beta_temp, beta_do, process_noise_sd, obs_noise_var]
func DefaultRickerPriors() []Prior {
	return []Prior{
		&TruncatedNormalPrior{Mu: 0.5, Sigma: 1.0, Lo: -2.0, Hi: 5.0}, // growth_rate
		&LogNormalPrior{Mu: 0.5, Sigma: 1.5},                           // density_dependence
		&TruncatedNormalPrior{Mu: 0, Sigma: 0.3, Lo: -2.0, Hi: 2.0},   // beta_flow
		&TruncatedNormalPrior{Mu: 0, Sigma: 0.3, Lo: -2.0, Hi: 2.0},   // beta_temp
		&TruncatedNormalPrior{Mu: 0, Sigma: 0.3, Lo: -2.0, Hi: 2.0},   // beta_do
		&HalfNormalPrior{Sigma: 0.5},                                    // process_noise_sd
		&LogNormalPrior{Mu: -1.5, Sigma: 1.0},                          // obs_noise_var
	}
}
