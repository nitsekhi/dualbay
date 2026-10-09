"""SQLite data store: schema creation, idempotent loaders, and reads as DataFrames.

Why SQLite and not Postgres/DuckDB: one file, no server, ships with Python, fine for
~10^6 rows. Everything is UPSERT-style, so re-running a loader never duplicates rows.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger(__name__)
DEFAULT_PATH = Path("data/dualbay.sqlite")
SCHEMA = Path(__file__).with_name("schema.sql")


@contextmanager
def connect(path: str | Path = DEFAULT_PATH):
    """Open (and initialise) the store. Commits on success, rolls back on error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    try:
        con.executescript(SCHEMA.read_text(encoding="utf-8"))
        yield con
        con.commit()
    except Exception:
        con.rollback()
        log.exception("database operation failed, rolled back")
        raise
    finally:
        con.close()


def _utc(ts) -> str:
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------- dimensions from config ----------

def load_dimensions(con: sqlite3.Connection) -> None:
    """Stations, zones and sensors from config/*.yaml."""
    from . import study_area

    for zid, z in study_area.zones().items():
        x0, y0, x1, y1 = z["geometry"].bounds
        con.execute("INSERT OR REPLACE INTO zone VALUES (?,?,?,?,?,?,?)",
                    (zid, z["bay"], z["description"], x0, y0, x1, y1))
    for sid, s in config.load("stations")["stations"].items():
        con.execute("INSERT OR REPLACE INTO station VALUES (?,?,?,?,?,?,?,?,?)",
                    (sid, s.get("name_ja"), s.get("name_en"), s.get("operator"), s["bay"], s["lat"], s["lon"],
                     int(bool(s.get("coord_verified", False))), study_area.zone_of(s["lon"], s["lat"])))
    for sid, s in config.load("sensors")["sensors"].items():
        con.execute("INSERT OR REPLACE INTO sensor VALUES (?,?,?,?)",
                    (sid, s["name"], s["provider"], s.get("resolution_m")))


# ---------- facts ----------

def load_insitu(con: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Long table from dualbay.insitu → insitu_obs. Returns rows inserted (duplicates ignored)."""
    known = {r[0] for r in con.execute("SELECT station_id FROM station")}
    unknown = set(df["station"]) - known
    if unknown:
        raise ValueError(f"Stations not in config/stations.yaml: {sorted(unknown)}")
    rows = [(r.station, _utc(r.time), r.layer if isinstance(r.layer, str) else None,
             None if pd.isna(r.depth_m) else float(r.depth_m), r.param, float(r.value), r.unit, r.qc, r.source_file)
            for r in df.itertuples(index=False)]
    before = con.total_changes
    con.executemany("INSERT OR IGNORE INTO insitu_obs(station_id,time_utc,layer,depth_m,param,value,unit,qc,source_file) "
                    "VALUES (?,?,?,?,?,?,?,?,?)", rows)
    return con.total_changes - before


def load_scenes(con: sqlite3.Connection, df: pd.DataFrame) -> int:
    rows = [(r.scene_id, r.sensor, _utc(r.time), None if pd.isna(r.cloud_pct) else float(r.cloud_pct),
             *[None if pd.isna(v) else float(v) for v in (r.lon_min, r.lat_min, r.lon_max, r.lat_max)])
            for r in df.itertuples(index=False)]
    before = con.total_changes
    con.executemany("INSERT OR IGNORE INTO scene(scene_id,sensor_id,time_utc,cloud_pct,lon_min,lat_min,lon_max,lat_max) "
                    "VALUES (?,?,?,?,?,?,?,?)", rows)
    return con.total_changes - before


def load_zone_stats(con: sqlite3.Connection, zone_df: pd.DataFrame, sensor_id: str = "pace_oci") -> int:
    """zone_spectra.csv (one row per band) → one row per scene×zone with the budget columns."""
    cols = ["n_water", "n_good", "frac_good", "frac_absaer", "frac_lowlw", "frac_straylight", "frac_higlint",
            "frac_turbidw", "frac_atmwarn", "frac_coastz", "frac_cldice", "frac_neg_412", "frac_neg_443"]
    d = zone_df.drop_duplicates(["granule", "zone"]).copy()
    if "n_good" not in d:
        d["n_good"] = d["n_valid"]
    for c in cols:
        if c not in d:
            d[c] = None
    # make sure the scenes exist
    sc = d.drop_duplicates("granule")[["granule", "time"]]
    con.executemany("INSERT OR IGNORE INTO scene(scene_id,sensor_id,time_utc) VALUES (?,?,?)",
                    [(g, sensor_id, _utc(t)) for g, t in zip(sc["granule"], sc["time"])])
    before = con.total_changes
    con.executemany(f"INSERT OR REPLACE INTO zone_scene_stats(scene_id,zone_id,{','.join(cols)}) VALUES (?,?,{','.join('?'*len(cols))})",
                    [(r["granule"], r["zone"], *[None if pd.isna(r[c]) else float(r[c]) for c in cols]) for _, r in d.iterrows()])
    return con.total_changes - before


def load_station_values(con: sqlite3.Connection, st_df: pd.DataFrame, sensor_id: str = "pace_oci") -> int:
    """station_values.csv → pixel_extract (spectral rows) and the chlor_a product row."""
    sc = st_df.drop_duplicates("granule")[["granule", "time"]]
    con.executemany("INSERT OR IGNORE INTO scene(scene_id,sensor_id,time_utc) VALUES (?,?,?)",
                    [(g, sensor_id, _utc(t)) for g, t in zip(sc["granule"], sc["time"])])
    rows = []
    for r in st_df.itertuples(index=False):
        wl = None if pd.isna(r.wavelength) else float(r.wavelength)
        rrs = None if pd.isna(r.rrs) else float(r.rrs)
        unc = getattr(r, "rrs_unc", None)
        unc = None if unc is None or pd.isna(unc) else float(unc)
        rows.append((r.granule, r.station, wl, rrs, unc, None, None, float(r.dist_km), int(r.n_valid), None, int(bool(r.passed))))
    chl = st_df.drop_duplicates(["granule", "station"])
    for r in chl.itertuples(index=False):
        if not pd.isna(r.chlor_a):
            rows.append((r.granule, r.station, None, None, None, "chlor_a", float(r.chlor_a), float(r.dist_km), int(r.n_valid), None, int(bool(r.passed))))
    before = con.total_changes
    con.executemany("INSERT OR REPLACE INTO pixel_extract(scene_id,station_id,wavelength,rrs,rrs_unc,product,product_val,dist_km,n_valid,window_cv,passed) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    return con.total_changes - before


def load_matchups(con: sqlite3.Connection, m: pd.DataFrame) -> int:
    rows = [(r.scene_id, r.station, r.param, _utc(r.insitu_time), float(r.dt_min), float(r.value), r.unit)
            for r in m.itertuples(index=False)]
    before = con.total_changes
    con.executemany("INSERT OR IGNORE INTO matchup(scene_id,station_id,param,insitu_time,dt_min,insitu_value,unit) VALUES (?,?,?,?,?,?,?)", rows)
    return con.total_changes - before


def new_run(con: sqlite3.Connection, model: str, param: str, tier: int | None = None, train_desc: str = "", cfg: dict | None = None) -> str:
    run_id = f"{model}-{param}-{uuid.uuid4().hex[:8]}"
    con.execute("INSERT INTO model_run VALUES (?,?,?,?,?,?,?)",
                (run_id, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), model, tier, param, train_desc, json.dumps(cfg or {})))
    return run_id


def save_predictions(con: sqlite3.Connection, run_id: str, matchup_ids, y_pred, y_lo=None, y_hi=None) -> None:
    n = len(matchup_ids)
    y_lo = [None] * n if y_lo is None else y_lo
    y_hi = [None] * n if y_hi is None else y_hi
    con.executemany("INSERT OR REPLACE INTO prediction VALUES (?,?,?,?,?)",
                    [(run_id, int(i), float(p), None if l is None else float(l), None if h is None else float(h))
                     for i, p, l, h in zip(matchup_ids, y_pred, y_lo, y_hi)])


def save_metrics(con: sqlite3.Connection, run_id: str, subset: str, m: dict) -> None:
    con.execute("INSERT OR REPLACE INTO metric VALUES (?,?,?,?,?,?,?,?,?,?)",
                (run_id, subset, int(m.get("n", 0)), m.get("rmse"), m.get("mae"), m.get("bias"), m.get("mape"),
                 m.get("mdsa"), m.get("r2"), m.get("picp68")))


# ---------- reads ----------

def read(con: sqlite3.Connection, sql: str, params=()) -> pd.DataFrame:
    return pd.read_sql_query(sql, con, params=params)


def matchup_spectra(con: sqlite3.Connection, param: str = "chl") -> pd.DataFrame:
    """Analysis table: one row per matchup with the station spectrum as wide columns rrs_<nm>.

    Only matchups whose pixel window passed QC are returned.
    """
    base = read(con, """
        SELECT v.matchup_id, v.scene_id, v.station_id, v.bay, v.zone_id, v.scene_time, v.insitu_value AS y, v.owt, v.split
        FROM v_matchup_wide v WHERE v.param = ?""", (param,))
    if base.empty:
        return base
    spec = read(con, """
        SELECT p.scene_id, p.station_id, p.wavelength, p.rrs FROM pixel_extract p
        WHERE p.wavelength IS NOT NULL AND p.passed = 1""")
    if spec.empty:
        return base.iloc[0:0]
    wide = spec.pivot_table(index=["scene_id", "station_id"], columns="wavelength", values="rrs")
    wide.columns = [f"rrs_{int(round(c))}" for c in wide.columns]
    out = base.merge(wide.reset_index(), on=["scene_id", "station_id"], how="inner")
    out["scene_time"] = pd.to_datetime(out["scene_time"], utc=True)
    return out
