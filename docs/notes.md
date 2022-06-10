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

The approximate master equation for the proposed stochastic simulation, which is inspired by the famous [Lotka-Volterra system](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations), takes the form

$$
\begin{align}
\frac{\delta}{\delta t} P(\dots, n_{ai}, \dots, t) + {\cal A}_{\delta t}\big[ P\big] &= \sum_{\forall i}\Lambda_{0i}(n_{0i}-1, n_{1i}, \dots )P(\dots, n_{0i}-1, \dots, t) \\
&+ \sum_{\forall i}\sum_{\forall a} (n_{ai}+1)\mu_{ai}P(\dots, n_{ai}+1, \dots, t) \\
&+ \sum_{\forall i}(n_{0i}-1)\alpha_{0i}\sum_{\forall i' \, {\sf prey}}\sum_{\forall a'} n_{a'i'}P(\dots, n_{0i}-1, n_{a'i'}, \dots, t)  \\
&+ \sum_{\forall i}\sum_{\forall a} (n_{ai}+1)\beta_{i} \sum_{\forall i' \, {\sf pred}} \sum_{\forall a'} n_{a'i'} P(\dots, n_{ai}+1, n_{a'i'}, \dots, t)  \\
&+ \sum_{\forall i}\sum_{\forall a} (n_{ai}+1) \gamma_{ai} P(\dots, n_{ai}+1, \dots, t) \\
&- \sum_{\forall i}\bigg[ \Lambda_{0i}(n_{0i}, n_{1i}, \dots ) + \sum_{\forall a}n_{ai}\mu_{ai} + n_{0i}\alpha_{0i} \sum_{\forall i' \, {\sf prey}}\sum_{\forall a'} n_{a'i'}  \\
& \qquad \quad + \sum_{\forall a}n_{ai}\beta_{ai} \sum_{\forall i' \, {\sf pred}}\sum_{\forall a'} n_{a'i'} + \sum_{\forall a}n_{ai}\gamma_{ai} \bigg] P(\dots, n_{ai}, \dots,t) \,,
\end{align}
$$

where the time $t$ is defined in units of years, and where: $\Lambda_{0i}(n_{0i}, n_{1i}, \dots ) = \tilde{\Lambda_{0i}}\sum_{\forall a > a^{\sf mat}_i}n_{ai}e^{-\lambda_i(n_{ai}-1)}$ is the density-dependent birth rate (which is only non-zero for age $a=0$ and given the typical reproductive maturity age for the $i$-th species $a^{\sf mat}_i$); $\mu_{ai}$ is the death rate for each age group of the species; $\alpha_{0i}$ is the increase in the baseline birth rate per fish caused by the increase in prey population; $\beta_{i}$ is the rate per fish of predation for all age groups of the species; and $\gamma_{ai}$ accounts for the rate of recreational fishing per fish in each age group of the species. Note that the ageing operator term on the LHS of the equation above is an approximation of a continuous derivative in age which takes into account the ageing between groups within a species. This term could take the rough form of 

$$
{\cal A}_{\delta t}\big[ P\big] \simeq \sum_{\forall i}\sum_{\forall a} \big[ P(\dots, n_{ai}, \dots,t) - P(\dots, n_{ai}+n^*_{ai}, n_{(a+1)i}-n^*_{ai}, \dots, t)\big] \,,
$$

where $n^*_{ai}(\delta t) = {\rm floor}(n_{ai} \delta t /\delta a)$ denotes the number of fish of the $i$-th species and age group $a$ ageing out of their group (where all groups have a width of $\delta a$ in the same time units as $t$). Note that in a stochastic algorithm, it makes more sense for this term to be modelled as a deterministic transition.

Look into the likelihood...

$$
\begin{align}
{\sf Likelihood} &= \sum_{{\sf data}}{\rm NB}\big[{\sf data};w_{i,{\sf survey}}\langle n_i(a_i,t_{{\sf data}})\rangle,k_{i,{\sf survey}}\big] \,,
\end{align}
$$
