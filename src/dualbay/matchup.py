"""Satellite–in-situ matchups.

1. `find_matchups` pairs each scene with in-situ observations at stations inside the scene
   footprint within ±window hours (default from config/sensors.yaml).
2. `window_stats` applies the pixel-window criteria of Bailey & Werdell (2006, RSE 102:12):
   at least half of the window valid, filtered mean within 1.5 σ, coefficient of variation ≤ 0.15.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

SURFACE_LAYERS = {None, "surface", "upper"}


def stations_frame(stations_cfg: dict | None = None) -> pd.DataFrame:
    cfg = stations_cfg or config.load("stations")
    rows = [{"station": sid, "lat": s["lat"], "lon": s["lon"], "bay": s["bay"], "coord_verified": s["coord_verified"]}
            for sid, s in cfg["stations"].items()]
    return pd.DataFrame(rows)


def surface_only(insitu: pd.DataFrame, max_depth_m: float = 2.0) -> pd.DataFrame:
    """Keep observations representative of the layer a satellite sees: surface/upper layer or depth ≤ max_depth_m."""
    layer_ok = insitu["layer"].isin(SURFACE_LAYERS - {None}) | insitu["layer"].isna()
    depth_ok = insitu["depth_m"].isna() | (insitu["depth_m"] <= max_depth_m)
    return insitu[layer_ok & depth_ok & (insitu["qc"] == "ok")]


def find_matchups(
    insitu: pd.DataFrame,
    scenes: pd.DataFrame,
    stations: pd.DataFrame | None = None,
    window_h: float | None = None,
    params: list[str] | None = None,
) -> pd.DataFrame:
    """Nearest-in-time in-situ value per (scene, station, param) within ±window_h hours.

    insitu: long table from dualbay.insitu (station ids must match config/stations.yaml).
    scenes: table from dualbay.inventory (time, bbox columns).
    """
    window_h = window_h if window_h is not None else config.load("sensors")["matchup_window_hours"]
    stations = stations if stations is not None else stations_frame()
    obs = surface_only(insitu)
    if params:
        obs = obs[obs["param"].isin(params)]
    obs = obs.assign(time_utc=pd.to_datetime(obs["time"]).dt.tz_convert("UTC"))
    scenes = scenes.assign(time=pd.to_datetime(scenes["time"], utc=True))
    win = pd.Timedelta(hours=window_h)

    out = []
    for sc in scenes.itertuples(index=False):
        has_box = not any(pd.isna(v) for v in (sc.lon_min, sc.lat_min, sc.lon_max, sc.lat_max))
        inside = stations
        if has_box:
            inside = stations[stations["lon"].between(sc.lon_min, sc.lon_max) & stations["lat"].between(sc.lat_min, sc.lat_max)]
        if inside.empty:
            continue
        near = obs[obs["station"].isin(inside["station"]) & (obs["time_utc"] - sc.time).abs().le(win)]
        if near.empty:
            continue
        near = near.assign(dt_min=(near["time_utc"] - sc.time).dt.total_seconds() / 60)
        best = near.loc[near["dt_min"].abs().groupby([near["station"], near["param"]]).idxmin()]
        for r in best.itertuples(index=False):
            out.append(
                {
                    "sensor": sc.sensor, "scene_id": sc.scene_id, "scene_time": sc.time,
                    "station": r.station, "param": r.param, "insitu_time": r.time_utc,
                    "dt_min": round(r.dt_min, 1), "value": r.value, "unit": r.unit,
                }
            )
    res = pd.DataFrame(out)
    if res.empty:
        return res
    return res.merge(stations[["station", "bay", "lat", "lon", "coord_verified"]], on="station", how="left")


def summary(matchups: pd.DataFrame) -> pd.DataFrame:
    """Matchup counts per sensor × bay × param — the core table of the data availability report."""
    if matchups.empty:
        return pd.DataFrame(columns=["sensor", "bay", "param", "n_matchups", "n_scenes", "n_stations"])
    g = matchups.groupby(["sensor", "bay", "param"])
    return g.agg(
        n_matchups=("value", "size"), n_scenes=("scene_id", "nunique"), n_stations=("station", "nunique")
    ).reset_index()


def window_stats(arr: np.ndarray, row: int, col: int, size: int = 3, max_cv: float = 0.15) -> dict:
    """Bailey & Werdell (2006) window statistics around (row, col) of a 2-D array (NaN = invalid).

    Returns filtered mean/median, CV, number of valid pixels and whether the window passes.
    """
    half = size // 2
    r0, r1 = max(row - half, 0), min(row + half + 1, arr.shape[0])
    c0, c1 = max(col - half, 0), min(col + half + 1, arr.shape[1])
    win = np.asarray(arr[r0:r1, c0:c1], dtype=float)
    valid = win[np.isfinite(win)]
    n_total = size * size
    res = {"n_valid": int(valid.size), "n_total": n_total, "mean": np.nan, "median": np.nan, "cv": np.nan, "passed": False}
    if valid.size < n_total / 2 + 0.5 * (n_total % 2):  # at least half of the window (5 of 9)
        return res
    med, sd = np.median(valid), np.std(valid)
    keep = valid[np.abs(valid - med) <= 1.5 * sd] if sd > 0 else valid
    fmean = float(np.mean(keep))
    cv = float(np.std(keep) / fmean) if fmean != 0 else np.inf
    res.update(mean=fmean, median=float(np.median(keep)), cv=cv, passed=bool(cv <= max_cv))
    return res
