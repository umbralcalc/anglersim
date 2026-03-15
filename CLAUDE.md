# anglersim — Project Context

Evaluating the impact of sustainability policies for fishing on freshwater fish populations in the UK using data-driven stochastic simulations. Built on the [stochadex](https://github.com/umbralcalc/stochadex) SDK.

## Project Structure

```
pkg/data/          # Data access and preparation
  counts.go         # Site-level count queries from raw CSV
  sites.go          # Sites CSV reader
  types.go          # Data types CSV reader
  coverage.go       # Coverage analysis (GetCoverageReport)
  panel.go          # Data prep pipeline (BuildPanel, LoadSiteTimeSeries)
  hydrology.go      # EA Hydrology API client (flow stations, daily readings)
  water_quality.go  # EA Water Quality API client (temp, DO, ammonia, BOD)
pkg/population/     # Stochastic population dynamics models
  ricker.go          # Stochastic Ricker iteration (log-density space)
pkg/inference/      # Bayesian inference
  priors.go          # Prior distributions (Uniform, TruncatedNormal, HalfNormal, LogNormal)
  smc.go             # SMC iterations + RunSMC() using EmbeddedSimulationRunIteration
  synthetic.go       # GenerateSyntheticData for testing parameter recovery
cmd/fit/            # CLI: random-search MLE (baseline)
cmd/fetchcovariates/ # CLI: fetch EA flow/WQ data for panel sites
cfg/                # Stochadex YAML configs
  smc_inference.yaml # Reference config for SMC with embedded inner sim (N=2)
nbs/                # GoNB Jupyter notebooks
  data_exploration.ipynb  # Data exploration and coverage analysis
dat/                # Data files (gitignored, see below)
```

## Data Files (dat/, gitignored)

Download NFPD bulk CSVs from https://environment.data.gov.uk/ecology/explorer/downloads/:
```bash
cd dat && for f in FW_Fish_Counts FW_Fish_Individual_Lengths FW_Fish_Banded_Measurements \
  FW_Fish_Bulk_Measurements FW_Fish_Data_Types FW_Fish_Sites; do \
  curl -sL "https://environment.data.gov.uk/ecology/explorer/downloads/$f.zip" -o "$f.zip" \
  && unzip -o "$f.zip" && rm "$f.zip"; done
```

Key files and sizes:
- `FW_Fish_Counts.csv` (148MB, 367k rows) — core dataset: species counts per survey
- `FW_Fish_Individual_Lengths.csv` (301MB, 7.5M rows) — individual fish measurements
- `FW_Fish_Sites.csv` (1.3MB, 16.8k rows) — site metadata with eastings/northings
- `FW_Fish_Data_Types.csv`, `FW_Fish_Banded_Measurements.csv`, `FW_Fish_Bulk_Measurements.csv`

Generated panel data (from notebook or BuildPanel):
- `brown_trout_panel.csv` — clean site×year panel for brown/sea trout
- `brown_trout_sites.csv` — site summary with coverage stats

## NFPD Data Nuances

**Important things to know about the data:**

- **NFPD is electrofishing survey data, NOT angling catch data.** These are scientific monitoring surveys conducted by the Environment Agency. Counts reflect survey effort and catchability, not total population.
- **Counts CSV columns:** DATA_OWNER, COUNTRY, LOCATION_NAME, REGION, AREA, SITE_ID, SITE_NAME, SURVEY_ID, EVENT_DATE, EVENT_DATE_YEAR, SURVEY_METHOD, SURVEY_STRATEGY, NO_OF_RUNS, SPECIES_NAME, LATIN_NAME, RUN1-RUN6, ALL_RUNS, SPCSNO (pop estimate), SURVEY_AREA, FISHED_AREA, ZERO_CATCH, etc.
- **Zero-catch rows** have `ZERO_CATCH=Yes` with empty species fields — they mean the site was surveyed but nothing was caught. Species absence (site surveyed, species not found) must be inferred by checking which species DON'T appear in a survey's records.
- **Survey methods vary.** ~93% of brown trout records use electric fishing (PDC, DC, AC). Filter with `ElectrofishingOnly: true` in PanelConfig to exclude seine netting, trapping, etc.
- **SPCSNO (population estimate)** only available for ~34% of brown trout records. Use `ALL_RUNS` (raw count) or density (`ALL_RUNS / FISHED_AREA`) as the primary metric.
- **Multiple surveys per site-year** are rare (~3.5% of site-years) but the panel pipeline sums them.
- **Survey area varies** between surveys at the same site, so density (fish/m²) is more comparable than raw count.
- **COVID gap:** Very few surveys in 2020 (1,173 records vs ~10k+ normally).
- **Best species for modelling:** Brown/sea trout (41k records, 794 sites with 10+ years electrofishing), then Bullhead, Stone loach, Minnow, Roach.

## GoNB Notebook Conventions

Notebooks use the GoNB Jupyter kernel (https://github.com/janpfeifer/gonb):
- Each cell is an independent main program — **variables do not persist between cells**. Every cell that needs data must load/compute it itself.
- First cell should set: `!*go mod edit -replace "github.com/umbralcalc/anglersim=/path/to/anglersim"`
- Imports go above `%%`, executable code goes below `%%`.
- For plotting, use `analysis.NewScatterPlotFromDataFrame` or `analysis.NewLinePlotFromDataFrame` from stochadex/pkg/analysis with `gonb_echarts.Display()`. No bar plot function exists.
- The stochadex analysis package provides: `NewScatterPlotFromDataFrame`, `NewScatterPlotFromPartition`, `NewLinePlotFromDataFrame`, `NewLinePlotFromPartition` (all use go-echarts).

## Integrated EA APIs

- **EA Hydrology API** (`environment.data.gov.uk/hydrology/`): Daily mean river flow (m³/s). Implemented in `pkg/data/hydrology.go`. Finds nearest flow station by easting/northing, fetches daily readings in yearly chunks, aggregates annual stats (mean, min, max, Q10, Q90).
- **EA Water Quality API** (`environment.data.gov.uk/water-quality/`): Temperature (°C), dissolved oxygen (mg/l), ammonia (mg/l), BOD (mg/l). Implemented in `pkg/data/water_quality.go`. Finds nearest river sampling point (type F6), fetches observations via CSV format (limit 2000/page), aggregates annual stats. Determinand codes: temp=0076, DO=9924, NH3=0111, BOD=0085.

Both APIs: no auth required, spatial search by lat/long + radius (km), matched to panel sites via nearest-point by easting/northing distance.

## Supplementary Data Sources (not yet integrated)

- **Rod Catch Returns:** Salmon & sea trout angling catch by river. GOV.UK downloads (ODS/CSV). Links to NFPD by river/catchment name. https://www.gov.uk/government/collections/salmonid-and-freshwater-fisheries-statistics
- **NRFA (river flow):** Daily mean flows from ~1,500 stations. `nrfa.ceh.ac.uk`. Links via spatial proximity. (EA Hydrology API now provides similar data.)
- **Fish stocking records:** Not public — requires FOI request to EA.
- **Policy/regulation timeline:** No machine-readable source — needs manual curation from legislation.gov.uk.

## SMC Inference Architecture

The SMC (Sequential Monte Carlo) inference uses the stochadex `EmbeddedSimulationRunIteration` to nest an inner particle-evaluation simulation inside an outer importance-sampling loop.

### Outer simulation (3 partitions)

| Partition | Iteration | State | Description |
|-----------|-----------|-------|-------------|
| `smc_proposals` | `SMCProposalIteration` | N×d flat | Draws particles from prior (round 1) or posterior proposal (subsequent rounds) |
| `smc_sim` | `EmbeddedSimulationRunIteration` | concatenated inner states | Runs inner sim to evaluate all N particles through the data |
| `smc_posterior` | `SMCPosteriorIteration` | mean(d) + cov(d²) + logZ(1) | Computes importance-weighted posterior statistics |

### Inner simulation (2 + 2N partitions, wrapped by embedded sim)

| Partition | Iteration | Description |
|-----------|-----------|-------------|
| `observed_data` | `FromStorageIteration` | Streams observed log-density data |
| `covariates` | `FromStorageIteration` | Streams environmental covariates (flow, temp, DO) |
| `ricker_p` (×N) | `RickerIteration` | Stochastic Ricker dynamics for particle p |
| `loglike_p` (×N) | `DataComparisonIteration` | Cumulative log-likelihood for particle p |

### Parameter forwarding

Particle parameters are forwarded from the outer proposal to inner partitions via **indexed `params_from_upstream`** on the embedded sim partition. Each particle p's 7 parameters (at flat offset p×7) are routed to the corresponding `ricker_p` and `loglike_p` partitions:

- `ricker_p/growth_rate` ← proposal[p×7 + 0]
- `ricker_p/density_dependence` ← proposal[p×7 + 1]
- `ricker_p/covariate_coefficients` ← proposal[p×7 + 2..4]
- `ricker_p/process_noise_sd` ← proposal[p×7 + 5]
- `loglike_p/variance` ← proposal[p×7 + 6] (obs_noise_var)

Log-likelihoods are extracted from the embedded output via indexed upstream on the posterior partition.

### Key functions

- `RunSMC(siteData, config)` — programmatic entry point, builds everything and returns `*SMCResult`
- `buildInnerSimulation(siteData, N)` — constructs inner sim settings/implementations
- `buildEmbeddedParamsFromUpstream(N, nParams)` — generates indexed upstream wiring
- `NewNormalDataComparison()` — helper to construct `DataComparisonIteration` without import alias conflicts

### YAML config

`cfg/smc_inference.yaml` is a reference config using the stochadex `embedded` field. The `smc_sim` main partition has no `iteration` — it's matched by name to the `embedded` run, which the API automatically wraps with `EmbeddedSimulationRunIteration`. Uses N=2 for readability; use `RunSMC()` for larger N.

### Model parameters (d=7)

| Index | Name | Prior | Description |
|-------|------|-------|-------------|
| 0 | `growth_rate` | TruncNorm(0.5, 1.0, -2, 5) | Baseline intrinsic growth rate |
| 1 | `density_dependence` | LogNorm(0.5, 1.5) | Strength of density-dependent mortality |
| 2 | `beta_flow` | TruncNorm(0, 0.3, -2, 2) | River flow covariate effect |
| 3 | `beta_temp` | TruncNorm(0, 0.3, -2, 2) | Water temperature covariate effect |
| 4 | `beta_do` | TruncNorm(0, 0.3, -2, 2) | Dissolved oxygen covariate effect |
| 5 | `process_noise_sd` | HalfNorm(0.5) | Process noise standard deviation |
| 6 | `obs_noise_var` | LogNorm(-1.5, 1.0) | Observation noise variance |

## Progress (as of March 2026)

1. ✅ Data downloaded and explored
2. ✅ Coverage analysis complete
3. ✅ Brown trout panel pipeline built (pkg/data/panel.go)
4. ✅ EA Hydrology API client built (flow stations, daily readings, annual stats)
5. ✅ EA Water Quality API client built (temp, DO, ammonia, BOD)
6. ✅ Flow + WQ data fetched and cached, joined to panel (cmd/fetchcovariates)
7. ✅ Stochastic Ricker population model (pkg/population/ricker.go)
8. ✅ SMC Bayesian inference using EmbeddedSimulationRunIteration (pkg/inference/smc.go)
9. ✅ YAML config updated to use stochadex `embedded` field (cfg/smc_inference.yaml)
10. Next: Multi-site batch fitting, hierarchical model across sites
11. Next: Model validation — held-out prediction, residual diagnostics
12. Next: Policy simulation scenarios with fitted model

---

# Stochadex SDK — Conventions

This project uses the [stochadex](https://github.com/umbralcalc/stochadex) SDK to build and run simulations.

## The Iteration Interface

Every simulation component implements `simulator.Iteration`:

```go
type Iteration interface {
    Configure(partitionIndex int, settings *Settings)
    Iterate(params *Params, partitionIndex int, stateHistories []*StateHistory,
            timestepsHistory *CumulativeTimestepsHistory) []float64
}
```

**Rules:**
- `Configure` is called once at setup. Use it to seed RNGs, read fixed config, allocate buffers. All mutable state must be re-initializable here (no statefulness residues between runs).
- `Iterate` is called each step. It must NOT mutate `params`. It returns the next state as `[]float64` with length equal to `StateWidth`.
- `stateHistories` gives access to all partitions' rolling state windows. `stateHistories[i].Values.At(row, col)` where row=0 is the latest state.
- `timestepsHistory.Values.AtVec(0)` is the current time. `timestepsHistory.NextIncrement` is the upcoming time step.
- Partitions communicate by wiring one partition's output state into another's params via `params_from_upstream` in config.

## YAML Config Format (API Code-Generation Path)

Simulations are defined in YAML and run via the stochadex CLI, which generates and executes Go code.

```yaml
main:
  partitions:
  - name: my_partition              # unique name
    iteration: myVar                # references a variable from extra_vars
    params:
      some_param: [1.0, 2.0]       # all param values are []float64
    params_from_upstream:           # wire upstream partition output → this partition's params
      latest_values:
        upstream: other_partition   # name of the upstream partition
        indices: [0, 1]            # optional: extract specific indices from upstream state
    params_as_partitions:           # reference partition names as param values (resolved to indices)
      data_partition: [some_partition]
    init_state_values: [0.0, 0.0]  # initial state (determines state_width)
    state_history_depth: 1          # rolling window size
    seed: 1234                      # RNG seed (0 = no randomness needed)
    extra_packages:                 # Go import paths
    - github.com/umbralcalc/stochadex/pkg/continuous
    extra_vars:                     # Go variable declarations
    - myVar: "&continuous.WienerProcessIteration{}"

  # A partition with no iteration field is matched by name to an embedded run.
  - name: my_embedded_sim
    params:
      init_time_value: [0.0]
      burn_in_steps: [0]
    params_from_upstream:
      inner_partition/param_name:   # "<innerPartition>/<paramName>" syntax
        upstream: source_partition  # forwards values to inner partition's params
        indices: [0]               # optional index selection

  simulation:
    output_condition: "&simulator.EveryStepOutputCondition{}"
    output_function: "&simulator.StdoutOutputFunction{}"
    termination_condition: "&simulator.NumberOfStepsTerminationCondition{MaxNumberOfSteps: 100}"
    timestep_function: "&simulator.ConstantTimestepFunction{Stepsize: 1.0}"
    init_time_value: 0.0

# Embedded simulation runs — matched to main partitions by name.
# The API wraps each with EmbeddedSimulationRunIteration automatically.
# Output state is the concatenation of all inner partition final states.
embedded:
- name: my_embedded_sim
  partitions:
  - name: inner_partition
    iteration: innerIter
    params: {}
    init_state_values: [0.0]
    state_history_depth: 2
    seed: 1
  simulation:
    output_condition: "&simulator.NilOutputCondition{}"
    output_function: "&simulator.NilOutputFunction{}"
    termination_condition: "&simulator.NumberOfStepsTerminationCondition{MaxNumberOfSteps: 10}"
    timestep_function: "&simulator.ConstantTimestepFunction{Stepsize: 1.0}"
    init_time_value: 0.0
```

### Common Output Functions
- `&simulator.StdoutOutputFunction{}` — print to stdout
- `simulator.NewJsonLogOutputFunction("./data.log")` — write JSON log file
- `&simulator.NilOutputFunction{}` — no output (for embedded sims)

### Common Termination Conditions
- `&simulator.NumberOfStepsTerminationCondition{MaxNumberOfSteps: N}`
- `&simulator.TimeElapsedTerminationCondition{MaxTimeElapsed: T}`

### Common Timestep Functions
- `&simulator.ConstantTimestepFunction{Stepsize: 0.1}`
- `&simulator.ExponentialDistributionTimestepFunction{RateLambda: 1.0}`

## Build & Run

```bash
go build ./...                                    # compile this project
go test -count=1 ./...                            # run all tests
go run github.com/umbralcalc/stochadex/cmd/stochadex --config cfg/builtin_example.yaml
```

## Testing Conventions

- Unit tests live alongside source as `*_test.go` files.
- Use `t.Run("description", func(t *testing.T) { ... })` subtests.
- Always include a subtest using `simulator.RunWithHarnesses(settings, implementations)` — this checks for NaN outputs, wrong state widths, params mutation, state history integrity, and statefulness residues.
- Load settings from a colocated YAML file (e.g., `my_iteration_settings.yaml` next to `my_iteration_test.go`).
- Use `gonum.org/v1/gonum/floats` for float comparisons, never raw `==`.
- No mocking — use real implementations.

## Built-In Iterations Reference

### continuous (github.com/umbralcalc/stochadex/pkg/continuous)

| Iteration | Params | Description |
|-----------|--------|-------------|
| `WienerProcessIteration` | `variances` | Brownian motion |
| `GeometricBrownianMotionIteration` | `variances` | Multiplicative Brownian motion |
| `OrnsteinUhlenbeckIteration` | `thetas`, `mus`, `sigmas` | Mean-reverting process |
| `DriftDiffusionIteration` | `drift_coefficients`, `diffusion_coefficients` | General drift-diffusion SDE |
| `DriftJumpDiffusionIteration` | `drift_coefficients`, `diffusion_coefficients`, `jump_rates` | Drift-diffusion with Poisson jumps |
| `CompoundPoissonProcessIteration` | `rates` | Compound Poisson process |
| `GradientDescentIteration` | `gradient`, `learning_rate`, `ascent` (optional) | Gradient-based optimization |
| `CumulativeTimeIteration` | (none) | Outputs cumulative simulation time |

### discrete (github.com/umbralcalc/stochadex/pkg/discrete)

| Iteration | Params | Description |
|-----------|--------|-------------|
| `PoissonProcessIteration` | `rates` | Poisson counting process |
| `BernoulliProcessIteration` | `state_value_observation_probs` | Binary outcomes |
| `BinomialObservationProcessIteration` | `observed_values`, `state_value_observation_probs`, `state_value_observation_indices` | Binomial draws |
| `CoxProcessIteration` | `rates` | Doubly-stochastic Poisson |
| `HawkesProcessIteration` | `intensity` | Self-exciting point process |
| `HawkesProcessIntensityIteration` | `background_rates` | Hawkes intensity function |
| `CategoricalStateTransitionIteration` | `transitions_from_N`, `transition_rates` | State machine |

### general (github.com/umbralcalc/stochadex/pkg/general)

| Iteration | Params | Description |
|-----------|--------|-------------|
| `ConstantValuesIteration` | (none) | Returns unchanged initial state |
| `CopyValuesIteration` | `partitions`, `partition_state_values` | Copies values from other partitions |
| `ParamValuesIteration` | `param_values` | Injects param values as state |
| `ValuesFunctionIteration` | (varies by Function) | User-defined function of state |
| `CumulativeIteration` | (none, wraps another) | Accumulates wrapped iteration output |
| `FromStorageIteration` | (none, uses Data field) | Streams pre-computed data |
| `FromHistoryIteration` | `latest_data_values` | Replays state history data |
| `EmbeddedSimulationRunIteration` | (varies) | Runs nested simulation each step |
| `ValuesCollectionIteration` | `values_state_width`, `empty_value` | Rolling collection of values |
| `ValuesChangingEventsIteration` | `default_values` | Routes by event value |
| `ValuesWeightedResamplingIteration` | `log_weight_partitions`, `data_values_partitions`, `past_discounting_factor` | Weighted resampling |
| `ValuesFunctionVectorMeanIteration` | `data_values_partition`, `latest_data_values` | Kernel-weighted rolling mean |
| `ValuesFunctionVectorCovarianceIteration` | `data_values_partition`, `latest_data_values`, `mean` | Kernel-weighted rolling covariance |

### inference (github.com/umbralcalc/stochadex/pkg/inference)

| Iteration | Params | Description |
|-----------|--------|-------------|
| `DataGenerationIteration` | `steps_per_resample`, `correlation_with_previous` (optional) | Synthetic data generation |
| `DataComparisonIteration` | `latest_data_values`, `cumulative`, `burn_in_steps` | Log-likelihood evaluation |
| `PosteriorMeanIteration` | `loglike_partitions`, `param_partitions`, `posterior_log_normalisation` | Posterior mean estimation |
| `PosteriorCovarianceIteration` | `loglike_partitions`, `param_partitions`, `posterior_log_normalisation`, `mean` | Posterior covariance estimation |
| `PosteriorLogNormalisationIteration` | `loglike_partitions`, `past_discounting_factor` | Log-normalisation tracking |

### kernels (github.com/umbralcalc/stochadex/pkg/kernels)

Kernels are not iterations — they implement `IntegrationKernel` and are used by iterations like `ValuesFunctionVectorMeanIteration`. Available: `ConstantIntegrationKernel`, `ExponentialIntegrationKernel`, `PeriodicIntegrationKernel`, `GaussianStateIntegrationKernel`, `TDistributionStateIntegrationKernel`, `BinnedIntegrationKernel`, `ProductIntegrationKernel`, `InstantaneousIntegrationKernel`.
