## Notes

**To dos on the data side here:**

1. Code up negative binomial MCMC to obtain the $\ln \langle n_i(a,t)\rangle$ values for each species as above.
2. Code up fitting a Gaussian correlated posterior profile to the $\ln \langle n_i(a,t)\rangle$ posterior for each species. 
3. Get new data on species death rates and recreational freshwater angling!!!

**IMPORTANT Data information to be aware of:**

- The surveys seem to [usually be conducted with electrofishing](https://environmentagency.blog.gov.uk/2016/05/11/how-we-carry-out-fish-surveys/) - which actually avoids harming the fish so that they can return to their population.
- `'Is Species Selective'` - this survey was focussed on a particular subset of species and so should not be used to get a representative sample of the whole population.
- `'Survey method'` - each method is distinct and so will not only give different k values for the negative binomial but also different means!!!

**To dos on the simulation side here:**

1. Neaten up these notes!
2. Code up a simulation of the above master equation for fish populations - should probably actually use the Gillespie algorithm this time based on the population sizes and the fact that there probably won't be much time dependence used beyond yearly discrete jumps in parameter values.
3. Obtain the maximum a posteriori (MAP) calibration of the full sim using the fitted Gaussian posterior profile from 2., any necessary Gaussian priors over $\ln \alpha$ and $\ln \beta$ values to stabilize the fitting and a Gaussian over the population count (which is reduced by Bernoulli trials corresponding to the survey) with variance equal to the tolerance $\epsilon$ which must be adaptively reduced as the fit improves.
4. Sampling algorithm to obtain the MAP of the simulation should alternate between exploration (localised sampling with a variance and fitting an overall lognormal profile to the posterior parameters) and exploitation using the profile to compute approximate gradients for the optimiser to direct the next regions for sampling.
5. Cross-validation with future data to verify the predictivity of the simulation.
6. Investigate how each fish species weight and length as a function of age evolves through time in the raw data and then build and fit a predictive model into the simulator which evolves this distribution for predators according to how much prey they receive - i.e., the $\alpha_i(a_i)$ term should increase the size of predators.
7. Look into the regional clustering cross-correlations between the mean counts of species between regions in order to potentially see patterns from common populations. This should hopefully be reflected in a spatial analysis.

**NEED TO FIRST GET AS MUCH DATA AS POSSIBLE ON RECREATIONAL FRESHWATER ANGLING!!!!**

The goal here is to calibrate a fully-stochastic causal model which evolves the fish counts, weights, lengths and ages for each species in each area.

The master equation for the proposed stochastic simulation, which is inspired by the famous [Lotka-Volterra system](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations), takes the form

**NEED TO REWRITE THIS EQUATION SO THAT THE BIRTH RATES ARE DENSITY-DEPENDENT + DEPEND ON THE SUM OVER COUNTS IN THE WHOLE POPULATION WHICH IS ABOVE REPRODUCTIVE AGE IN EACH SPECIES' CASE + THERE NEED TO BE COUPLING TERMS BETWEEN THE AGE GROUPS - MIGHT DECIDE THAT IT'S EASIER TO MAKE THIS COMPONENT DETERMINISTIC? (ONLY A SLIGHT REFACTOR OF THE CODE CURRENTLY)**

$$
\begin{align}
\bigg[ \frac{\partial}{\partial a_i} + \frac{\partial}{\partial t}\bigg] P(\dots, n_i, \dots, a_i, t) &= \sum_{\forall i}\Lambda_i(a_i)P(\dots, n_i-1, \dots, a_i, t) \\
&+ \sum_{\forall i} (n_i+1)\mu_i(a_i)P(\dots, n_i+1, \dots, a_i, t) \\
&+ \sum_{\forall i} \alpha_i(a_i)(n_i-1)\sum_{\forall i' \, {\sf prey}} n_{i'}P(\dots, n_i-1, n_{i'}, \dots, a_i, t)  \\
&+ \sum_{\forall i} \beta_i(n_i+1) \sum_{\forall i' \, {\sf pred}} n_{i'}P(\dots, n_i+1, n_{i'}, \dots, a_i, t)  \\
&+ \sum_{\forall i} \gamma_i(a_i)(n_i+1)P(\dots, n_i+1, \dots,  a_i, t) \\
&- \sum_{\forall i}\bigg[ \Lambda_i(a_i) + n_i\mu_i(a_i) + \alpha_i(a_i)n_i \sum_{\forall i' \, {\sf prey}} n_{i'}  \\
& \qquad \quad + \beta_in_i \sum_{\forall i' \, {\sf pred}} n_{i'} + \gamma_i(a_i)n_i \bigg] P(\dots, n_i, \dots, a_i, t) \,,
\end{align}
$$

where the time $t$ is defined in units of years, and where: $\Lambda_i(a_i) = \Lambda_i\mathbb{1}_{a_i=0}$ is the birth rate (which is only non-zero for $a_i=0$); $\mu_i(a_i)$ is the age-dependent death rate; $\alpha_i(a_i) = \alpha_i\mathbb{1}_{a_i=0}$ is the increase in the baseline birth rate per fish caused by each predation event; $\beta_i$ is the rate per fish of predation; and $\gamma_i(a_i)$ accounts for the rate of recreational fishing per fish for the $i$-th species.

One could choose to calibrate the stochastic model using the data by using the following 'deterministic' model which can be derived by taking the first moment of the above master equation with respect to $n_i$, yielding

$$
\begin{align}
\bigg[ \frac{\partial}{\partial a_i} + \frac{\partial}{\partial t}\bigg] \langle n_i(a_i,t)\rangle &= \Lambda_i(a_i) - \mu_i(a_i)\langle n_i(a_i,t)\rangle + \alpha_i(a_i)\langle n_i(a_i,t)\rangle\sum_{\forall i' \, {\sf prey}} \langle n_{i'}(t)\rangle \\
& - \beta_i\langle n_i(a_i,t)\rangle\sum_{\forall i' \, {\sf pred}} \langle n_{i'}(t)\rangle - \gamma_i(a_i) \langle n_i(a_i,t)\rangle \\
{\sf Likelihood} &= \sum_{{\sf data}}{\rm NB}\big[{\sf data};w_{i,{\sf survey}}\langle n_i(a_i,t_{{\sf data}})\rangle,k_{i,{\sf survey}}\big] \,,
\end{align}
$$

but this would not include population size-based fluctuations (which are probably fairly small in most cases due to the large population numbers). To include these fluctuations with limited additional computational cost, one could perform a Van Kampen large-$N$ expansion about the population mean (see, e.g., appendix C of [this paper](https://www.sciencedirect.com/science/article/pii/S1755436521000013)) but let's do something a bit more computationally fancy with the full simulation instead, just for fun. 