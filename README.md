# dualbay

Water quality retrieval and digital twin framework for **Tokyo Bay** and **Sagami Bay**.
Research project at Yokohama National University (Higa Lab), 2026–2029.

This first version covers the kenkyusei-period groundwork (Oct–Nov 2026):
study area zones, in-situ data parsing, satellite scene inventory, and matchup counting —
the inputs for the **data availability report** that decides the main sensor.

## Setup

```bash
# with uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,inventory]"

# or with pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,inventory]"

pytest            # 4 tests, should pass
```

For PACE searches, put your NASA Earthdata login in `~/.netrc`:
```
machine urs.earthdata.nasa.gov login YOUR_USER password YOUR_PASSWORD
```

## Workflow

```bash
# 1. Study area zones → data/processed/study_area.geojson
dualbay study-area
#    with a land mask (OSM land polygons or GSHHG as GeoJSON, clipped to the region):
dualbay study-area --coastline data/raw/land.geojson

# 2. In-situ: download CSV/Excel files by hand into data/raw/insitu/, then
dualbay parse-insitu data/raw/insitu/*.csv
#    one file per station, named by the station id from config/stations.yaml,
#    or pass --station kawasaki_island for a single file

# 3. Scene inventory over both bays (Sentinel-2/3 via Copernicus STAC, PACE via Earthdata)
dualbay inventory --start 2024-03-01 --end 2026-09-30 --sensors s3_olci pace_oci
#    SGLI / PRISMA / HISUI: export the scene list from the portal and load it with
#    dualbay.inventory.load_manual_list()

# 4. Matchups (±3 h, surface layer, QC ok) and the summary table
dualbay matchups --insitu data/processed/insitu.csv --scenes data/processed/scenes.csv
```

`data/processed/matchup_summary.csv` gives matchups per sensor × bay × parameter —
the core table of the data availability report.

## PACE first look (week of 7 Oct 2026)

Three figures for the lab meeting: clear-scene availability per month, Rrs spectra per zone,
and the standard NASA chlorophyll at monitoring posts.

```bash
pip install -e ".[pace]"

# quick test: 5 granules
dualbay pace --start 2025-07-01 --end 2025-08-31 --limit 5

# spectra for one year (Rrs + Rrs_unc, PACE_OCI_L2_AOP v3.1) — each granule is downloaded to a temp dir, the bay subset read, the file deleted
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

## Full chain (after the PACE first look)

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

Docs: `docs/ARCHITECTURE.md` (data flow, design decisions, error handling, known limits),
`docs/METHODS.md` (report-ready methods text). Dashboard: load `data/processed/powerbi/*.csv`
into Power BI, create the relationships listed at the top of `measures.dax`, paste the measures.

## Layout

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
  pace.py              PACE OCI L2: remote subset reading, zone spectra, pixel budget, station values
  flags.py             NASA l2_flags: standard mask, per-zone flag fractions (ABSAER, LOWLW, ...)
  figures.py           first-look figures from the PACE tables
  db.py / schema.sql   SQLite star schema and idempotent loaders
  baseline.py          OCx, CI, 3-band, NDCI, Nechad, Dogliotti; local recalibration; metrics incl. MdSA, PICP
  owt.py               optical water types (k-means, silhouette), novelty flag, coverage table
  models.py            random-forest retrieval with per-pixel 68 % interval and calibration check
  transfer.py          temporal holdout, static vs adaptive, spatial transfer, learning curve
  lag.py               surface Chl-a vs bottom DO lags, hypoxia events, lead time, blue-tide index
  export.py            Power BI tables + DAX measures
  demo.py              synthetic end-to-end run
  cli.py               `dualbay` command
tests/                 12 tests: parser, matchups, PACE reader, flags, store, baselines, OWT, models, transfer, lag, export
data/                  raw/ and processed/ (git-ignored)
```

## Before real matchups

- **Station coordinates** in `config/stations.yaml` are approximate (`coord_verified: false`).
  Replace them with official values from the MLIT Kanto Regional Bureau monitoring reports
  or the data files. For a 300 m OLCI pixel, a 1 km error already matters.
- **Header keywords**: if `parse-insitu` reports "No parameter columns recognised", add the
  file's header words to `config/insitu_columns.yaml`.
- PACE OCI L2 AOP v3.1 Rrs is a *provisional* product (NASA: biased low ~0.001 sr⁻¹ vs AERONET-OC; largest errors at 400–413 nm in complex waters). Record `product_version` and never mix v3.1 with later reprocessings in one analysis.
- ABSAER and ATMWARN pixels are kept (reported per zone), because absorbing-aerosol errors over Tokyo Bay are the research object (Higa et al. 2024). Use `pace.MASK_EXTRA = ("ABSAER", "ATMWARN")` for a conservative run.
- The in-situ parser keeps only surface/upper-layer values (or depth ≤ 2 m) for matchups;
  bottom-layer DO stays in the table for the hypoxia work but is not paired with satellite data.

## Next modules (planned)

`preprocess/` ACOLITE runs and masks · `baseline/` OCx, CI, 3-band, NDCI, Nechad, Dogliotti ·
`owt/` optical water types · `models/` PLSR, RF, GP, PINN · `dt/` assimilation and forecast.
