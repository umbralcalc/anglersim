# anglersim

Evaluating the impact of sustainability policies on freshwater fish populations in the UK using data-driven simulations.

This project was built using the [stochadex framework](https://github.com/umbralcalc/stochadex).

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

### Integrated environmental covariates

| Source | Contains | Access | Integration |
|--------|----------|--------|-------------|
| **EA Hydrology API** | Daily mean river flow (m³/s) from gauging stations | `environment.data.gov.uk/hydrology/` | Nearest station by easting/northing, annual stats (mean, min, max, Q10, Q90) joined to panel |
| **EA Water Quality API** | Temperature (°C), dissolved oxygen (mg/l), ammonia, BOD | `environment.data.gov.uk/water-quality/` | Nearest river sampling point (type F6), annual stats joined to panel |

Both APIs require no authentication. Fetched data is cached in `dat/hydrology/` and `dat/water_quality/`. Run `cmd/fetchcovariates` to populate.

### Supplementary data sources (not yet integrated)

| Source | Contains | Access | Link to NFPD |
|--------|----------|--------|-------------|
| **Rod Catch Returns** | Annual salmon & sea trout angling catch by river | GOV.UK statistics downloads (ODS/CSV) | By river/catchment name |
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

**6. Policy scenario simulation** — Monte Carlo forward projections under predefined policy scenarios (baseline, climate +1/+2 C, low abstraction, drought, water quality improvement, combined). Pass `--regional` to produce a per-region summary CSV.

```bash
go run ./cmd/simulate \
  --panel dat/brown_trout_panel_with_covariates.csv \
  --params dat/hierarchical_results.csv \
  --out dat/projections.csv \
  --summary dat/projections_summary.csv \
  --regional dat/regional_summary.csv \
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

## Key results

### Fleet-level scenario impacts (20yr projection, 790 sites)

| Scenario | Med density change | Sites declining | Critical (>50% loss) | Mean extinction P |
|----------|-------------------|-----------------|---------------------|-------------------|
| baseline | -8.6% | 59.4% | 17.6% | 0.059 |
| climate +1°C | -11.6% | 59.5% | 19.4% | 0.063 |
| climate +2°C | -14.2% | 61.3% | 22.8% | 0.073 |
| low abstraction (+15% flow) | -8.4% | 58.2% | 17.5% | 0.059 |
| drought (-25% flow) | -10.3% | 60.6% | 17.7% | 0.059 |
| water quality (+15% DO) | -8.8% | 59.4% | 17.6% | 0.059 |
| combined +2°C & +15% DO | -14.2% | 61.3% | 22.8% | 0.074 |

The model includes an Allee effect (depensatory growth at low density), which prevents unrealistic recovery projections for near-zero populations. Even the baseline scenario projects a -8.6% median decline, consistent with observed downward trends in the survey data. **Temperature remains the dominant driver**, adding ~6pp of additional decline under +2°C warming. Flow and dissolved oxygen interventions remain negligible at the population level.

### Regional vulnerability (climate +2°C scenario)

| Region | Sites | Med density change | Critical (>50% loss) | Mean extinction P |
|--------|-------|-------------------|---------------------|-------------------|
| Kent & South London | 7 | -79.0% | 71.4% | 0.263 |
| West | 34 | -33.4% | 38.2% | 0.139 |
| Wessex | 37 | -32.6% | 40.5% | 0.127 |
| Eastern | 39 | -31.1% | 41.0% | 0.116 |
| Northern | 16 | -24.4% | 31.2% | 0.108 |
| North | 153 | -17.2% | 19.0% | 0.047 |
| Yorkshire | 128 | -14.8% | 21.1% | 0.125 |
| Devon & Cornwall | 206 | -4.6% | 16.5% | 0.021 |
| North East | 71 | -5.5% | 19.7% | 0.088 |

Kent & South London (7 sites) faces the most severe projected decline. The West, Wessex, and Eastern regions show >30% median losses with 38-41% of sites in critical decline. Devon & Cornwall — the largest region — is the most resilient with only -4.6% median change under +2°C.

### Seasonal covariate experiment

Seasonal aggregation was tested as an alternative to annual means: summer mean flow (Jun–Sep), summer mean temp (Jun–Sep), and winter mean DO (Dec–Feb). The seasonal covariates did not improve the flow/DO signal (effects remained ~0.01) and the temperature effect became slightly positive — implying warming *reduces* decline, which is ecologically implausible for brown trout. This confirms that the annual mean temperature signal is primarily capturing correlated long-term trends (warming + other changes) rather than direct thermal stress. The model retains annual means as covariates; seasonal aggregation functions remain available in the codebase (`AggregateSeasonalFlow`, `AggregateSeasonalWQ`) and the panel CSV includes seasonal columns for future use.

### Rod catch correlation analysis

Rod catch returns (GOV.UK supplementary data tables, 2008–2024) were matched to NFPD electrofishing sites to test whether angling harvest correlates with population trends. 255 NFPD sites were matched to 46 rod catch rivers using river names extracted from NFPD site parent names. Results across 1,421 site-year observations:

- **Pooled correlation:** r=0.14 (driven by river-size confounding — bigger rivers have both more anglers and more fish)
- **Within-site correlation:** median Spearman r=0.007; 50/50 positive vs negative; 7% significant at p<0.05 (chance level)
- **First-differenced:** r=0.03, p=0.39 — year-over-year changes in rod catch do not predict changes in density
- **Lagged (harvest → next year):** r=0.04, p=0.27 — no depletion signal

Rod catch returns reflect angler effort and access rather than site-level fishing pressure. With >90% catch-and-release in recent years, actual harvest is a negligible fraction of reported catch. Rod catch is not incorporated as a model covariate.

### Caveats

- Near-zero flow/DO effects persist under both annual and seasonal covariate resolution, suggesting the limitation is fundamental to the data (sparse WQ sampling, covariate station matching noise) rather than fixable by temporal aggregation
- Model fitted on density (fish/m²) from electrofishing surveys — habitat loss wouldn't show up
- Temperature signal likely captures correlated time trends (warming + other long-term changes) rather than direct thermal stress — confirmed by seasonal experiment where summer-specific temps lost the signal
- Regions with few sites (< 20) should be interpreted with caution
- The Allee effect prior (LogNormal) is weakly informative — the strength of depensatory growth is estimated per site but may be poorly identified at sites with limited low-density observations

## Data sources for the project

This is an entirely non-commercial project intended for research and learning purposes (see the [MIT License](LICENSE)). The data for this project has been collected from these sources:

- NFPD Bulk: [https://environment.data.gov.uk/ecology/explorer/downloads/](https://environment.data.gov.uk/ecology/explorer/downloads/)
- NFPD API: [https://environment.data.gov.uk/ecology/api/v1/index.html](https://environment.data.gov.uk/ecology/api/v1/index.html)
- Water Quality API: [https://environment.data.gov.uk/water-quality/](https://environment.data.gov.uk/water-quality/)
- National River Flow Archive: [https://nrfa.ceh.ac.uk/](https://nrfa.ceh.ac.uk/)
- Rod Catch Returns: [https://www.gov.uk/government/collections/salmonid-and-freshwater-fisheries-statistics](https://www.gov.uk/government/collections/salmonid-and-freshwater-fisheries-statistics)
