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

## Data sources for the project

This is an entirely non-commercial project intended for research and learning purposes (see the [MIT License](LICENSE)). The data for this project has been collected from these sources:

- NFPD Bulk: [https://environment.data.gov.uk/ecology/explorer/downloads/](https://environment.data.gov.uk/ecology/explorer/downloads/)
- NFPD API: [https://environment.data.gov.uk/ecology/api/v1/index.html](https://environment.data.gov.uk/ecology/api/v1/index.html)
- Water Quality API: [https://environment.data.gov.uk/water-quality/](https://environment.data.gov.uk/water-quality/)
- National River Flow Archive: [https://nrfa.ceh.ac.uk/](https://nrfa.ceh.ac.uk/)
- Rod Catch Returns: [https://www.gov.uk/government/collections/salmonid-and-freshwater-fisheries-statistics](https://www.gov.uk/government/collections/salmonid-and-freshwater-fisheries-statistics)
