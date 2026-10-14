# dualbay

Water quality retrieval and digital twin framework for **Tokyo Bay** and **Sagami Bay**
from PACE OCI hyperspectral imagery and in-situ monitoring data.
Research project at Yokohama National University (Higa Laboratory), 2026–2029.
Author: Lilya Javadova. License: MIT.

The package covers the whole path from raw data to results: study-area zones, PACE granule
reading, in-situ parsing, satellite–station matchups, a small database, baseline algorithms,
optical water types, a retrieval model with per-pixel uncertainty, transfer experiments,
lag analysis between surface chlorophyll and bottom oxygen, and an export for Power BI.
Every step runs offline on synthetic data (`dualbay demo`), so the chain is tested before
real data arrive.

## Repository map

| Path | What it holds |
|---|---|
| `src/dualbay/` | the analysis chain (module list below) |
| `config/` | study-area zones, sensors, stations, header keywords for in-situ files |
| `tests/` | 12 automated tests, all offline (`pytest`) |
| `docs/` | `ARCHITECTURE.md` (data flow, design decisions, limits), `METHODS.md` (methods text) |
| `prototype/` | 3D prototype of the twin on **synthetic data** |
| `data/` | raw and processed data (git-ignored) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev,inventory,pace]"
pytest                                                 # 12 tests, all offline
```

For PACE access, put your NASA Earthdata login in `~/.netrc` (never in the repository):
```
machine urs.earthdata.nasa.gov login YOUR_USER password YOUR_PASSWORD
```

## Workflow

```bash
# 1. Study-area zones → data/processed/study_area.geojson
dualbay study-area
#    with a land mask (OSM land polygons or GSHHG as GeoJSON, clipped to the region):
dualbay study-area --coastline data/raw/land.geojson

# 2. In-situ: download CSV/Excel files by hand into data/raw/insitu/, then
dualbay parse-insitu data/raw/insitu/*.csv
#    one file per station, named by the station id from config/stations.yaml,
#    or pass --station kawasaki_island for a single file

# 3. Scene inventory over both bays (Sentinel-2/3 via Copernicus STAC, PACE via Earthdata)
dualbay inventory --start 2024-03-01 --end 2026-09-30 --sensors s3_olci pace_oci

# 4. Matchups (±3 h, surface layer, QC ok) and the summary table
dualbay matchups --insitu data/processed/insitu.csv --scenes data/processed/scenes.csv
```

## PACE OCI

```bash
# quick test: 5 granules
dualbay pace --start 2025-07-01 --end 2025-08-31 --limit 5

# reflectance for one year (Rrs + Rrs_unc, PACE_OCI_L2_AOP) —
# each granule is downloaded to a temporary folder, the bay subset is read, the file is deleted
dualbay pace --start 2025-01-01 --end 2025-12-31 --max-cloud 70

# standard chlorophyll (PACE_OCI_L2_BGC) for the same period
dualbay pace --product bgc --start 2025-01-01 --end 2025-12-31 --max-cloud 70

# figures → figures/01_pace_availability.png, 02_pace_spectra.png, 03_pace_station_chl.png
dualbay figures
```

Tables land in `data/processed/pace/`; progress is saved every 10 granules, so an interrupted run
keeps what it already read. Each figure also gets a CSV with the numbers behind it.
`--max-cloud` filters on the cloud cover of the whole granule (a wide swath), not of the bay;
the per-zone clear fraction in `zone_spectra.csv` is the real measure.

## Analysis chain

```bash
# 1. put everything into the store (idempotent; re-run any time)
dualbay load --insitu data/processed/insitu.csv --scenes data/processed/scenes.csv \
             --zone-spectra data/processed/pace/zone_spectra.csv \
             --station-values data/processed/pace/station_values.csv \
             --matchups data/processed/matchups.csv

# 2. baselines (tier 1–2), uncertainty model (tier 3), temporal + spatial transfer → results/
dualbay analyze --param chl --train-bay tokyo

# 3. RQ3 groundwork from in-situ alone: chl → bottom DO lags, hypoxia events, lead times
dualbay lag --insitu data/processed/insitu.csv

# 4. Power BI star schema + measures → data/processed/powerbi/
dualbay export

# 0. no data yet? run the whole chain on synthetic data (≈2 min) → data/demo/
dualbay demo
```

Dashboard: load `data/processed/powerbi/*.csv` into Power BI, create the relationships listed
at the top of `measures.dax`, paste the measures.

## Modules

```
config/
  study_area.yaml      zones: tokyo_inner, uraga_channel, miura_transition, sagami_coastal, sagami_open
  sensors.yaml         sensors, collections, matchup window
  stations.yaml        in-situ stations with coordinates
  insitu_columns.yaml  header keywords for Japanese/English in-situ exports
src/dualbay/
  study_area.py        zones, land clipping, point → zone
  insitu.py            CSV/Excel → long table (station, time JST, layer, depth, param, value, unit, qc)
  inventory.py         scene search (CDSE STAC, Earthdata) and manual scene lists
  matchup.py           scene–station pairing, summary, Bailey & Werdell (2006) window statistics
  pace.py              PACE OCI L2: download → bay subset → zone spectra, pixel budget, station values
  flags.py             NASA l2_flags: bit table from the file, standard mask, per-zone flag fractions
  figures.py           summary figures from the PACE tables
  db.py / schema.sql   SQLite star schema and idempotent loaders
  baseline.py          OCx, CI, 3-band, NDCI, Nechad, Dogliotti; local recalibration; accuracy metrics
  owt.py               optical water types (k-means, silhouette), novelty flag, coverage table
  models.py            random-forest retrieval with per-pixel 68 % interval and calibration check
  transfer.py          temporal holdout, static vs adaptive, spatial transfer, learning curve
  lag.py               surface Chl-a vs bottom DO lags, hypoxia events, lead time, blue-tide index
  export.py            Power BI tables + DAX measures
  demo.py              synthetic end-to-end run
  cli.py               the `dualbay` command
```

## Notes before real matchups

- **Station coordinates** in `config/stations.yaml` are approximate except Kemigawa
  (`coord_verified: false`). Replace them with official values from the station documents.
- **Header keywords**: if `parse-insitu` reports "No parameter columns recognised", add the
  file's header words to `config/insitu_columns.yaml`.
- **PACE product versions**: the L2 AOP product is still provisional and is reprocessed from time
  to time (v3.1 → v3.2 in 2026, Version 4 announced). `product_version` is recorded for every
  granule; never mix versions in one analysis.
- **Spectral range**: the standard L2 reflectance ends at 719 nm. Algorithms that need
  near-infrared bands (3-band, Dogliotti, Nechad-865) require Level-1B processing with ACOLITE.
- **Flags**: ABSAER (absorbing aerosol) pixels are kept and reported per zone because they are the
  research object over Tokyo Bay. Use `pace.MASK_EXTRA = ("ABSAER", "ATMWARN")` for a
  conservative run.
- The in-situ parser keeps only surface/upper-layer values (or depth ≤ 2 m) for matchups;
  bottom-layer DO stays in the table for the hypoxia work but is not paired with satellite data.
