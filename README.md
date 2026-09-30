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
  cli.py               `dualbay` command
tests/                 parser, matchups, window statistics, zones
data/                  raw/ and processed/ (git-ignored)
```

## Before real matchups

- **Station coordinates** in `config/stations.yaml` are approximate (`coord_verified: false`).
  Replace them with official values from the MLIT Kanto Regional Bureau monitoring reports
  or the data files. For a 300 m OLCI pixel, a 1 km error already matters.
- **Header keywords**: if `parse-insitu` reports "No parameter columns recognised", add the
  file's header words to `config/insitu_columns.yaml`.
- The in-situ parser keeps only surface/upper-layer values (or depth ≤ 2 m) for matchups;
  bottom-layer DO stays in the table for the hypoxia work but is not paired with satellite data.

## Next modules (planned)

`preprocess/` ACOLITE runs and masks · `baseline/` OCx, CI, 3-band, NDCI, Nechad, Dogliotti ·
`owt/` optical water types · `models/` PLSR, RF, GP, PINN · `dt/` assimilation and forecast.
