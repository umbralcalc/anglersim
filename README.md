# anglersim

A methodological study: can the UK's National Fish Population Database (NFPD) electrofishing surveys, combined with the EA Hydrology and Water Quality APIs, support a climate-driven population dynamics model for brown trout?

**Short answer: not at present.** The NFPD covers 1,372 brown trout sites with ≥10 years of density observations, but the EA Water Quality API matches only ~3% of those site-years (Hydrology does better at ~21%). After filtering to sites with ≥10 years of fully-covariated data, **19 sites** remain — too sparse to fit a meaningful hierarchical climate-effects model.

This repo documents what the pipeline does, where the data runs out, and what would be needed to do this properly. Built on the [stochadex framework](https://github.com/umbralcalc/stochadex).

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

The density-only side of the data is rich. Where it falls down is at the join with environmental covariates.

### Environmental covariates — the bottleneck

| Source | Channel | Panel-row coverage* | Sites with ≥10 yrs† |
|---|---|---|---|
| EA Hydrology API | Mean / Q10 / Q90 daily flow (m³/s) | **21.2%** | 124 |
| EA Water Quality API | Mean / max temperature (°C) | **3.4%** | 20 |
| EA Water Quality API | Mean / min dissolved oxygen (mg/l) | **3.2%** | 20 |
| **All three together** | flow + temp + DO | — | **19** |

*Among 13,293 non-zero-density brown trout panel rows. †Among the 790 sites with ≥10 years of density observations.

The reason is structural: the EA Hydrology gauging network covers most rivers, but the Water Quality F6 sampling-point network is much sparser. Even with nearest-station matching (`cmd/fetchcovariates`), most site-years are unmatchable to a real WQ measurement.

A climate-effects model trained on all three EA covariates therefore has ~19 effective sites. A flow-only model has ~124. The full 790-site fits previously published in this repo were obtained from an earlier version of the data loader that silently zero-filled missing covariate values — see *Historical artifacts* below — and should not be used as scientific results.

### Integrated environmental covariates

| Source | Contains | Access | Integration |
|--------|----------|--------|-------------|
| **EA Hydrology API** | Daily mean river flow (m³/s) from gauging stations | `environment.data.gov.uk/hydrology/` | Nearest station by easting/northing, annual + summer stats joined to panel |
| **EA Water Quality API** | Temperature (°C), dissolved oxygen (mg/l), ammonia, BOD | `environment.data.gov.uk/water-quality/` | Nearest river sampling point (type F6), annual + winter-DO / summer-temp stats joined to panel |

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

This is slow (API rate-limited) but caches results in `dat/hydrology/` and `dat/water_quality/`. The output marks rows with missing covariates as empty (not zero) — the panel loader (`pkg/data/panel.go`) now drops site-years with any required covariate missing, so downstream fits only see real measurements.

**3. Batch SMC inference** — fit the stochastic Ricker model independently to every qualifying site using Sequential Monte Carlo. With the patched loader, `--min-years` (default 10) now means "≥N years of fully-covariated data per site" — site-years with any required covariate missing have already been dropped by `LoadAllSiteTimeSeries`.

```bash
go run ./cmd/batchsmc \
  --panel dat/brown_trout_panel_with_covariates.csv \
  --out dat/batch_smc_results.csv \
  --workers 4 --particles 500 --rounds 3 --min-years 10
```

Produces per-site posterior means, standard deviations, and log marginal likelihoods. With the default filter and the current EA Water Quality coverage, this fits ~19 sites — not 790.

**4. Hierarchical empirical Bayes** — use the Stage 1 posteriors to estimate population-level hyperparameters for covariate effects, then re-fit all sites under hierarchical priors. With only ~19 eligible sites the pool is small; per-site posteriors dominate and the EB shrinkage is weak.

```bash
go run ./cmd/batchhierarchical \
  --stage1 dat/batch_smc_results.csv \
  --panel dat/brown_trout_panel_with_covariates.csv \
  --out dat/hierarchical_results.csv \
  --workers 4 --particles 500 --rounds 3 --iterations 3 --min-years 10
```

**5. Held-out validation** — fit on training data (all but the last N years), forward-simulate predictions, and compute RMSE/coverage metrics.

```bash
go run ./cmd/validate \
  --panel dat/brown_trout_panel_with_covariates.csv \
  --out dat/validation_results.csv \
  --preds dat/validation_predictions.csv \
  --workers 4 --holdout 3 --sims 200
```

**6. Policy scenario simulation** — Monte Carlo forward projections under predefined policy scenarios. Pass `--regional` to produce a per-region summary CSV. Magnitudes of climate-related effects should be interpreted in light of the very small eligible site count — see *Findings* below.

```bash
go run ./cmd/simulate \
  --panel dat/brown_trout_panel_with_covariates.csv \
  --params dat/hierarchical_results.csv \
  --out dat/projections.csv \
  --summary dat/projections_summary.csv \
  --regional dat/regional_summary.csv \
  --scenario all --horizon 20 --sims 500 --workers 4
```

**7. Visualise** — open `nbs/model_validation.ipynb` to plot parameter distributions, validation diagnostics, hierarchical vs independent comparisons, and policy scenario impacts.

### Single-site debugging

`cmd/fit` runs a quick random-search MLE on a single site — useful for debugging the model without the full SMC machinery:

```bash
go run ./cmd/fit \
  --panel dat/brown_trout_panel_with_covariates.csv \
  --site 1915 --n 5000
```

## Findings

### What the data can support

- **Per-site density dynamics.** The stochastic Ricker model with Allee depensation fits the 790 sites with ≥10 years of density observations. Held-out validation RMSE is wide (median 0.72 log-units across all 573 validated sites, equivalent to ~±100% in density) but the 90% prediction intervals cover most held-out observations (median 1.00, mean 0.82). The model captures equilibrium density and process noise reasonably well at the per-site level.
- **Flow-only effects on a 124-site subset** is feasible in principle, though the within-site signal is weak: the median per-site `beta_flow` posterior across the 124-site fits is +0.011 with a cross-site sd of 0.21 — consistent with noise.

### What the data cannot support

- **A climate-effects model.** Only 19 brown trout sites have ≥10 years of complete temperature + DO covariate coverage. That is too few sites for population-level inference about climate sensitivity, even before considering that linear `beta_temp × temp` is the wrong functional form for a thermal-optimum response.
- **Quantitative "+1°C / +2°C costs X% density" claims.** With essentially no within-site temperature signal at most sites — and only 19 sites with enough WQ coverage to contribute information — the temperature coefficient cannot be identified from these data. Any specific magnitude is noise.
- **Quantitative DO or BOD interventions.** Same coverage problem.

### Why the original 790-site climate fit was misleading

The earlier version of `pkg/data/panel.go` used a `parseFloatDefault` helper that returned `0.0` for any missing or unparseable covariate value, with no warning. Because 97.6% of brown trout panel rows have at least one missing covariate, the SMC was being trained on a panel where most temperature, flow and DO observations were literally the number zero. The per-site posteriors correctly returned ~noise (the likelihood was invariant to the covariate effects), but the iterative empirical Bayes step then pooled across all 790 noise estimates. Iterative shrinkage drove the inferred `sigma_beta_*` to the 0.01 floor, locking every site to the same `mu_beta_*`, which was itself residual sampling noise.

When projected forward over a 30-year horizon under climate scenarios, that locked-in `mu_beta_temp` produced visible but spurious population trajectories — including, depending on the random sign of the noise, the implausible "warming increases brown trout density" pattern seen in the most recent runs.

The loader has been patched to propagate `NaN` and the batch commands now drop incomplete site-years and filter on covariate coverage by default, so this can't silently happen again.

### Held-out validation

Median RMSE on a 3-year holdout is 0.72 log-units across 573 sites; mean 90% interval coverage is 0.82 (median 1.00). 76 of 573 sites have <50% coverage and are flagged in the notebook. The model is a useful population descriptor but most of the held-out predictive width comes from process noise; the validation does not distinguish a model with non-zero covariate effects from one with covariate effects pinned at zero.

### Rod catch correlation analysis (unchanged)

Rod catch returns (GOV.UK supplementary data tables, 2008–2024) were matched to NFPD electrofishing sites to test whether angling harvest correlates with population trends. 255 NFPD sites were matched to 46 rod catch rivers using river names extracted from NFPD site parent names. Results across 1,421 site-year observations:

- **Pooled correlation:** r=0.14 (driven by river-size confounding — bigger rivers have both more anglers and more fish)
- **Within-site correlation:** median Spearman r=0.007; 50/50 positive vs negative; 7% significant at p<0.05 (chance level)
- **First-differenced:** r=0.03, p=0.39 — year-over-year changes in rod catch do not predict changes in density
- **Lagged (harvest → next year):** r=0.04, p=0.27 — no depletion signal

Rod catch returns reflect angler effort and access rather than site-level fishing pressure. With >90% catch-and-release in recent years, actual harvest is a negligible fraction of reported catch. Rod catch is not incorporated as a model covariate.

## What would be needed to do this properly

1. **Better water-quality coverage.** The bottleneck is the EA F6 sampling network. Options include: FOI-route to EA internal water quality data, climate reanalysis grids (e.g. HadUK-Grid for temperature) to give a value for every site-year, or modelled-DO products from catchment-scale process models.
2. **Non-linear thermal response.** A linear `beta_temp × temp` term cannot represent a thermal optimum (cool → tolerable → lethal). With richer data, a quadratic term or a piecewise-linear above/below preferred-range model would be the right form.
3. **A design that breaks the warming-vs-recovery confound.** UK rivers recovered from industrial pollution over much of the panel period, so a naïve year-on-year fit will conflate "warming" with "everything else improved." A difference-in-differences design or a pre/post split on a known intervention would help.
4. **A richer set of sites for the hierarchical pool.** The current 19-site bottleneck is what makes empirical Bayes useless here. Loosening the species filter, or running a multi-species hierarchical model, would increase the pool.

## Historical artifacts in `dat/`

The CSVs currently checked into `dat/` (`batch_smc_results.csv`, `hierarchical_results.csv`, `projections.csv`, `projections_summary.csv`, `regional_summary.csv`, `validation_*.csv`) were produced by the original pipeline before the silent-zero-fill loader bug was identified and fixed. They cover all 790 sites — including the 663 sites that were "fitted" against all-zero covariates. They are kept in the repository for reference and reproducibility but **should not be cited as scientific results**.

The hyperparameter comment at the top of the old `hierarchical_results.csv` (`mu_temp=+0.006855 sigma_temp=0.014659 mu_do=-0.012116 sigma_do=0.013581`) is the visible signature of the runaway-shrinkage failure described above: tiny sigma, near-zero mu of arbitrary sign. Re-running the patched pipeline against the same panel produces a much smaller fit (~19 sites, no hierarchical pooling worth the name, no statistically meaningful climate coefficients).

## Caveats

- The model is fit on density (fish/m²) from electrofishing surveys — habitat loss would not show up as density change.
- The Allee effect prior (LogNormal) is weakly informative — strength of depensatory growth is estimated per site but may be poorly identified at sites with limited low-density observations.
- Regions with few sites (< 20) should be interpreted with caution.
- Seasonal aggregation (summer flow/temp, winter DO) is available in the panel CSV and `pkg/data/{hydrology,water_quality}.go` but does not materially change the coverage picture above.

## Data sources for the project

This is an entirely non-commercial project intended for research and learning purposes (see the [MIT License](LICENSE)). The data for this project has been collected from these sources:

- NFPD Bulk: [https://environment.data.gov.uk/ecology/explorer/downloads/](https://environment.data.gov.uk/ecology/explorer/downloads/)
- NFPD API: [https://environment.data.gov.uk/ecology/api/v1/index.html](https://environment.data.gov.uk/ecology/api/v1/index.html)
- Water Quality API: [https://environment.data.gov.uk/water-quality/](https://environment.data.gov.uk/water-quality/)
- National River Flow Archive: [https://nrfa.ceh.ac.uk/](https://nrfa.ceh.ac.uk/)
- Rod Catch Returns: [https://www.gov.uk/government/collections/salmonid-and-freshwater-fisheries-statistics](https://www.gov.uk/government/collections/salmonid-and-freshwater-fisheries-statistics)
