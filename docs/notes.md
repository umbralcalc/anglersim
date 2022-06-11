## Notes

The goal here is to calibrate a stochastic causal model which evolves the fish counts, weights, lengths and ages for each species in each area. To do this, we will combine some well-known models from mathematical ecology with supervised learning.

**To dos:**

1. Code up negative binomial MCMC to obtain the $\ln \langle n_i(t)\rangle$ values for each species as above.
2. Code up fitting a Gaussian correlated posterior profile to the $\ln \langle n_i(t)\rangle$ posterior for each species. 
3. Get new data on species death rates and recreational freshwater angling in order to put in reasonable parameters to the simulation.
4. Convert the current simulation to using 'realisations' in the matrix dimension that used to correspond to age bins.
5. Obtain the maximum a posteriori (MAP) calibration of the full sim using the fitted Gaussian posterior profile from 2., using any necessary Gaussian priors over $\ln \alpha$ and $\ln \beta$ values to stabilize the fitting and a Gaussian over the population count (which is reduced by Bernoulli trials corresponding to the survey) with variance equal to the tolerance $\epsilon$ which must be adaptively reduced as the fit improves.
6. Sampling algorithm to obtain the MAP of the simulation should alternate between exploration (localised sampling with a variance and fitting an overall lognormal profile to the posterior parameters) and exploitation using the profile to compute approximate gradients for the optimiser to direct the next regions for sampling.
7. Cross-validation with future data to verify the predictivity of the simulation.
8. Investigate how the ages of each species evolve over time (and whether or not these are correlated to the $\langle n_i(t)\rangle$ values obtained from the simulation at the relevant time period). 
9. Build a predictive supervised learning model for ages, weights and lengths.
10. Look into the regional clustering cross-correlations between the mean counts of species between regions in order to potentially see patterns from common populations. This should hopefully be reflected in a spatial analysis.


**IMPORTANT Data information to be aware of:**

- The surveys seem to [usually be conducted with electrofishing](https://environmentagency.blog.gov.uk/2016/05/11/how-we-carry-out-fish-surveys/) - which actually avoids harming the fish so that they can return to their population.
- `'Is Species Selective'` - this survey was focussed on a particular subset of species and so should not be used to get a representative sample of the whole population.
- `'Survey method'` - each method is distinct and so will not only give different k values for the negative binomial but also different means!!!

**NEED TO FIRST GET AS MUCH DATA AS POSSIBLE ON RECREATIONAL FRESHWATER ANGLING!!!!**

The approximate master equation for the proposed stochastic simulation, which is inspired by the famous [Lotka-Volterra system](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations), takes the form

$$
\begin{align}
\frac{{\rm d}}{{\rm d} t} P(\dots, n_{i}, \dots, t) &= \sum_{\forall i}\Lambda_{i}(n_{i}-1)P(\dots, n_{i}-1, \dots, t) \\
&+ \sum_{\forall i} (n_{i}+1)\mu_{i}P(\dots, n_{i}+1, \dots, t) \\
&+ \sum_{\forall i}(n_{i}-1)\alpha_{i}\sum_{\forall i' \, {\sf prey}}n_{i'}P(\dots, n_{i}-1, n_{i'}, \dots, t)  \\
&+ \sum_{\forall i} (n_{i}+1)\beta_{i} \sum_{\forall i' \, {\sf pred}} n_{i'} P(\dots, n_{i}+1, n_{i'}, \dots, t)  \\
&+ \sum_{\forall i} (n_{i}+1) \gamma_{i} P(\dots, n_{i}+1, \dots, t) \\
&- \sum_{\forall i}\bigg[ \Lambda_{i}(n_{i}) + n_{i}\mu_{i} + n_{i}\alpha_{i} \sum_{\forall i' \, {\sf prey}} n_{i'}  \\
& \qquad \quad + n_{i}\beta_{i} \sum_{\forall i' \, {\sf pred}} n_{i'} + n_{i}\gamma_{i} \bigg] P(\dots, n_{i}, \dots,t) \,,
\end{align}
$$

where the time $t$ is defined in units of years, and where: $\Lambda_{i}(n_{i}) = \tilde{\Lambda_{i}}n_{i}e^{-\lambda_i(n_{i}-1)}$ is the density-dependent birth rate; $\mu_{i}$ is the species death rate; $\alpha_{i}$ is the increase in the baseline birth rate per fish caused by the increase in prey population; $\beta_{i}$ is the rate per fish of predation of the species; and $\gamma_{i}$ accounts for the rate of recreational fishing per fish of the species.

Look into the likelihood...

$$
\begin{align}
{\sf Likelihood} &= \sum_{{\sf data}}{\rm NB}\big[{\sf data};w_{i,{\sf survey}}\langle n_i(t_{{\sf data}})\rangle,k_{i,{\sf survey}}\big] \,,
\end{align}
$$
