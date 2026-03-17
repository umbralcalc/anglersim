# anglersim

Evaluating the impact of sustainability policies for fishing on freshwater fish populations in the UK using data-driven simulations.

## Data assessment

### NFPD (National Fish Population Database)

The core dataset is the EA's electrofishing survey data (NFPD bulk downloads). Coverage summary:

- **367,505** count records across **16,703** unique sites and **126** species
- **Year range:** 1973-2025 (peak survey effort 2002-2016, ~10k-17k records/year)
- **1,538 sites** have 10+ distinct survey years (sufficient for individual site-level time series modelling)
- **4,370 sites** have 5+ survey years (suitable for hierarchical/pooled models)
- **Top species by records:** Brown/sea trout, Bullhead, Stone loach, Minnow, Roach, Atlantic salmon, Perch, European eel, Dace, Chub
- **Individual length measurements:** 7.5M records linked to surveys via SURVEY_ID, enabling size-structured population models
- **Geographic coverage:** All EA areas in England, strongest in North, Devon & Cornwall, Yorkshire

This is sufficient to build and train stochastic population dynamics models at well-surveyed sites.

### Supplementary data sources (to be integrated)

| Source | Contains | Access | Link to NFPD |
|--------|----------|--------|-------------|
| **Rod Catch Returns** | Annual salmon & sea trout angling catch by river | GOV.UK statistics downloads (ODS/CSV) | By river/catchment name |
| **Water Quality Archive** | Temperature, dissolved oxygen, ammonia, BOD, nutrients | EA API: `environment.data.gov.uk/water-quality/` | Spatial join on easting/northing |
| **National River Flow Archive** | Daily mean flows from ~1,500 stations, catchment descriptors | NRFA API: `nrfa.ceh.ac.uk` | Spatial proximity + river name |
| **EA Hydrology API** | 15-minute river level/flow readings | `environment.data.gov.uk/hydrology/` | Spatial join |
| **WFD Fish Classifications** | Ecological quality ratios, status classes | Catchment Data Explorer | WFD waterbody ID |

**Not publicly available:** Fish stocking records (EA internal, available via FOI) and a structured policy/regulation change timeline (requires manual curation from legislation.gov.uk and EA bylaws).

## Analysis pipeline

The analysis runs as a sequence of CLI binaries. Each step produces CSV files consumed by the next. The notebooks in `nbs/` visualise the outputs.

### Pipeline overview

```
data_exploration.ipynb          (1) Explore raw NFPD data, build panel
        |
        v
  brown_trout_panel.csv
  brown_trout_sites.csv
        |
        v
cmd/fetchcovariates             (2) Fetch EA flow & water quality data
        |
        v
  brown_trout_panel_with_covariates.csv
        |
        +-----------+-----------+
        |           |           |
        v           v           v
  cmd/batchsmc  cmd/validate  cmd/fit
      (3)          (5)       (debug)
        |
        v
  batch_smc_results.csv
        |
        v
  cmd/batchhierarchical         (4) Empirical Bayes refinement
        |
        v
  hierarchical_results.csv
        |
        v
  cmd/simulate                   (6) Policy scenario projections
        |
        v
  projections.csv
  projections_summary.csv
        |
        v
  model_validation.ipynb         (7) Visualise all results
```

### Step-by-step

**1. Build the panel** — run `nbs/data_exploration.ipynb` to produce `dat/brown_trout_panel.csv` and `dat/brown_trout_sites.csv` from the raw NFPD bulk downloads.

**2. Fetch covariates** — match panel sites to the nearest EA flow stations and water quality sampling points, fetch daily readings, and join annual statistics to the panel.

```bash
go run ./cmd/fetchcovariates \
  --sites dat/brown_trout_sites.csv \
  --panel dat/brown_trout_panel.csv \
  --out dat/brown_trout_panel_with_covariates.csv
```

This is slow (API rate-limited) but caches results in `dat/hydrology/` and `dat/water_quality/`.

**3. Batch SMC inference** — fit the stochastic Ricker model independently to every qualifying site using Sequential Monte Carlo.

```bash
go run ./cmd/batchsmc \
  --panel dat/brown_trout_panel_with_covariates.csv \
  --out dat/batch_smc_results.csv \
  --workers 4 --particles 500 --rounds 3
```

Produces per-site posterior means, standard deviations, and log marginal likelihoods.

**4. Hierarchical empirical Bayes** — use the Stage 1 posteriors to estimate population-level hyperparameters for covariate effects, then re-fit all sites under hierarchical priors. Iterates until convergence.

```bash
go run ./cmd/batchhierarchical \
  --stage1 dat/batch_smc_results.csv \
  --panel dat/brown_trout_panel_with_covariates.csv \
  --out dat/hierarchical_results.csv \
  --workers 4 --particles 500 --rounds 3 --iterations 3
```

**5. Held-out validation** — fit on training data (all but the last N years), forward-simulate predictions, and compute RMSE/coverage metrics.

```bash
go run ./cmd/validate \
  --panel dat/brown_trout_panel_with_covariates.csv \
  --out dat/validation_results.csv \
  --preds dat/validation_predictions.csv \
  --workers 4 --holdout 3 --sims 200
```

**6. Policy scenario simulation** — Monte Carlo forward projections under predefined policy scenarios (baseline, climate +1/+2 C, low abstraction, drought, water quality improvement, combined).

```bash
go run ./cmd/simulate \
  --panel dat/brown_trout_panel_with_covariates.csv \
  --params dat/hierarchical_results.csv \
  --out dat/projections.csv \
  --summary dat/projections_summary.csv \
  --scenario all --horizon 20 --sims 500 --workers 4
```

**7. Visualise** — open `nbs/model_validation.ipynb` to plot parameter distributions, validation diagnostics, hierarchical vs independent comparisons, and policy scenario impacts. The notebook expects all the above CSV files to exist in `dat/`.

### Single-site debugging

`cmd/fit` runs a quick random-search MLE on a single site — useful for debugging the model without the full SMC machinery:

```bash
go run ./cmd/fit \
  --panel dat/brown_trout_panel_with_covariates.csv \
  --site 1915 --n 5000
```

## Data sources for the project

This is an entirely non-commercial project intended for research and learning purposes (see the [MIT License](LICENSE)). The data for this project has been collected from these sources:

- NFPD Bulk: [https://environment.data.gov.uk/ecology/explorer/downloads/](https://environment.data.gov.uk/ecology/explorer/downloads/)
- NFPD API: [https://environment.data.gov.uk/ecology/api/v1/index.html](https://environment.data.gov.uk/ecology/api/v1/index.html)
- Water Quality API: [https://environment.data.gov.uk/water-quality/](https://environment.data.gov.uk/water-quality/)
- National River Flow Archive: [https://nrfa.ceh.ac.uk/](https://nrfa.ceh.ac.uk/)
- Rod Catch Returns: [https://www.gov.uk/government/collections/salmonid-and-freshwater-fisheries-statistics](https://www.gov.uk/government/collections/salmonid-and-freshwater-fisheries-statistics)
