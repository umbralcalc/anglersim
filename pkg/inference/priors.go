package inference

import (
	stdinf "github.com/umbralcalc/stochadex/pkg/inference"
)

// DefaultRickerPriors returns the default priors for the 8 Ricker model parameters:
// [growth_rate, density_dependence, beta_flow, beta_temp, beta_do, process_noise_sd, obs_noise_var, allee_effect]
func DefaultRickerPriors() []stdinf.Prior {
	return []stdinf.Prior{
		&stdinf.TruncatedNormalPrior{Mu: 0.5, Sigma: 1.0, Lo: -2.0, Hi: 5.0}, // growth_rate
		&stdinf.LogNormalPrior{Mu: 0.5, Sigma: 1.5},                           // density_dependence
		&stdinf.TruncatedNormalPrior{Mu: 0, Sigma: 0.3, Lo: -2.0, Hi: 2.0},   // beta_flow
		&stdinf.TruncatedNormalPrior{Mu: 0, Sigma: 0.3, Lo: -2.0, Hi: 2.0},   // beta_temp
		&stdinf.TruncatedNormalPrior{Mu: 0, Sigma: 0.3, Lo: -2.0, Hi: 2.0},   // beta_do
		&stdinf.HalfNormalPrior{Sigma: 0.5},                                    // process_noise_sd
		&stdinf.LogNormalPrior{Mu: -1.5, Sigma: 1.0},                          // obs_noise_var
		&stdinf.LogNormalPrior{Mu: 2.0, Sigma: 1.5},                           // allee_effect (gamma)
	}
}
