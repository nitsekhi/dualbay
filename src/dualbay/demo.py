"""Offline end-to-end run on synthetic data.

Purpose: prove every link of the chain (in-situ → scenes → extracts → matchups → OWT → baselines →
models with uncertainty → transfer experiments → lag analysis → store → Power BI export) without
network access, and give each module realistic-looking input until real data arrive.

Everything produced here is MOCK DATA and is written under data/demo/. Nothing from it may be
shown as a result.

Synthetic world
  Tokyo Bay stations: eutrophic, Chl-a lognormal around 15 µg/L with summer blooms, bottom DO that
  drops after blooms with a 7–12 day lag and summer stratification.
  Sagami stations: Chl-a around 1 µg/L, no hypoxia.
  Rrs: a 3-component bio-optical toy (water + phytoplankton absorption + particle backscatter) with
  noise, so retrieval algorithms have something physical to recover. Not a radiative-transfer model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import baseline, db, export, lag, matchup, models, owt, transfer

log = logging.getLogger(__name__)
WL = np.arange(400, 721, 10)

STATIONS = {  # extra demo stations for Sagami so transfer experiments have something to test
    "sagami_hiratsuka": dict(name_ja="平塚沖", name_en="Hiratsuka offshore (demo)", operator="demo", bay="sagami", lat=35.28, lon=139.35, coord_verified=False),
    "sagami_odawara": dict(name_ja="小田原沖", name_en="Odawara offshore (demo)", operator="demo", bay="sagami", lat=35.22, lon=139.18, coord_verified=False),
    "sagami_open1": dict(name_ja="相模湾中央", name_en="Sagami central (demo)", operator="demo", bay="sagami", lat=35.10, lon=139.35, coord_verified=False),
}


def toy_rrs(chl, tsm, cdom, rng):
    """Toy forward model: Rrs ≈ 0.05 * bb / (a + bb) with simple spectral shapes."""
    wl = WL[None, :]
    aw = 0.0045 * np.exp((wl - 400) / 110)                      # water absorption, rising to red
    aph = 0.03 * chl[:, None] ** 0.9 * (np.exp(-((wl - 440) / 35) ** 2) + 0.5 * np.exp(-((wl - 675) / 12) ** 2))
    acdom = cdom[:, None] * np.exp(-0.015 * (wl - 440))
    bbp = 0.004 * tsm[:, None] ** 0.8 * (550 / wl) ** 0.6
    bbw = 0.0012 * (500 / wl) ** 4.3
    rrs = 0.05 * (bbp + bbw) / (aw + aph + acdom + bbp + bbw)
    return rrs * rng.lognormal(0, 0.05, rrs.shape)


def build(out_dir: str | Path = "data/demo", years=(2024, 2025, 2026), seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dbp = out / "dualbay_demo.sqlite"
    if dbp.exists():
        dbp.unlink()

    stations = matchup.stations_frame()
    extra = pd.DataFrame([{"station": k, "lat": v["lat"], "lon": v["lon"], "bay": v["bay"], "coord_verified": False} for k, v in STATIONS.items()])
    stations = pd.concat([stations, extra], ignore_index=True)

    # ---- hourly in-situ for all stations
    days = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="D", tz="Asia/Tokyo")
    rows = []
    truth = []
    for st in stations.itertuples(index=False):
        tokyo = st.bay == "tokyo"
        doy = days.dayofyear.to_numpy()
        season = np.sin((doy - 100) / 365 * 2 * np.pi)                      # peaks in summer
        base_chl = (15 if tokyo else 1.0) * np.exp(0.6 * season)
        blooms = np.zeros(len(days))
        for _ in range(6 if tokyo else 1):
            c = rng.integers(120, 290); w = rng.integers(4, 10); amp = rng.uniform(1.5, 4)
            blooms += amp * np.exp(-((doy - c) / w) ** 2)
        chl = base_chl * (1 + blooms) * rng.lognormal(0, 0.25, len(days))
        tsm = (8 if tokyo else 1.5) * np.exp(0.3 * season) * rng.lognormal(0, 0.3, len(days))
        cdom = (0.6 if tokyo else 0.08) * rng.lognormal(0, 0.2, len(days))
        temp_s = 17 + 9 * season + rng.normal(0, 0.5, len(days))
        temp_b = 15 + 4 * season + rng.normal(0, 0.3, len(days))
        # bottom DO: baseline 7, minus stratification effect, minus lagged bloom effect (lag 9 d)
        lagged = np.roll(np.convolve(chl, np.ones(5) / 5, mode="same"), 9)
        do_b = 7.5 - 0.35 * np.clip(temp_s - temp_b, 0, None) - (0.09 if tokyo else 0.0) * lagged + rng.normal(0, 0.4, len(days))
        do_b = np.clip(do_b, 0.2, 9)
        for i, d in enumerate(days):
            for h in (3, 9, 15, 21):
                t = (d + pd.Timedelta(hours=h)).isoformat()
                rows += [(st.station, t, "surface", 0.5, "chl", chl[i] * rng.lognormal(0, 0.1), "ug/L"),
                         (st.station, t, "surface", 0.5, "turbidity", tsm[i] * 1.1 * rng.lognormal(0, 0.1), "FTU"),
                         (st.station, t, "surface", 0.5, "water_temp", temp_s[i], "degC")]
                if tokyo:
                    rows += [(st.station, t, "bottom", 12.0, "do", do_b[i], "mg/L"),
                             (st.station, t, "bottom", 12.0, "water_temp", temp_b[i], "degC")]
            truth.append((st.station, d, chl[i], tsm[i], cdom[i]))
    insitu = pd.DataFrame(rows, columns=["station", "time", "layer", "depth_m", "param", "value", "unit"])
    insitu["time"] = pd.to_datetime(insitu["time"])
    insitu["qc"] = "ok"
    insitu["source_file"] = "demo"
    truth = pd.DataFrame(truth, columns=["station", "date", "chl", "tsm", "cdom"])

    # ---- scenes: one PACE pass per day at ~03:30 UTC, clear with p=0.35 (summer lower)
    scenes, extracts = [], []
    for d in days:
        p_clear = 0.45 - 0.2 * (6 <= d.month <= 9)
        if rng.random() > p_clear:
            continue
        sid = f"DEMO_PACE.{d.strftime('%Y%m%d')}T033000"
        t_utc = (d.tz_convert("UTC") + pd.Timedelta(hours=3, minutes=30))
        scenes.append({"sensor": "pace_oci", "scene_id": sid, "time": t_utc, "cloud_pct": 100 * (1 - p_clear),
                       "lon_min": 139.0, "lat_min": 34.8, "lon_max": 140.3, "lat_max": 35.9})
        tr = truth[truth["date"] == d]
        rrs = toy_rrs(tr["chl"].to_numpy(), tr["tsm"].to_numpy(), tr["cdom"].to_numpy(), rng)
        # atmospheric-correction error: inner Tokyo Bay blue bands biased low (toy version of Higa 2024)
        for i, st in enumerate(tr["station"]):
            is_tokyo = stations.set_index("station").loc[st, "bay"] == "tokyo"
            spec = rrs[i].copy()
            if is_tokyo:
                spec[WL < 470] -= rng.uniform(0.0005, 0.0025)
            passed = rng.random() > 0.15
            for k, w in enumerate(WL):
                extracts.append({"time": t_utc, "granule": sid, "station": st, "dist_km": 0.4, "n_valid": 9 if passed else 3,
                                 "passed": passed, "wavelength": float(w), "rrs": float(spec[k]), "chlor_a": np.nan})
    scenes = pd.DataFrame(scenes)
    extracts = pd.DataFrame(extracts)

    # ---- matchups
    m = matchup.find_matchups(insitu, scenes, stations=stations, window_h=3, params=["chl", "turbidity"])

    # ---- store
    with db.connect(dbp) as con:
        db.load_dimensions(con)
        for k, v in STATIONS.items():
            con.execute("INSERT OR REPLACE INTO station VALUES (?,?,?,?,?,?,?,?,?)",
                        (k, v["name_ja"], v["name_en"], v["operator"], v["bay"], v["lat"], v["lon"], 0, None))
        db.load_insitu(con, insitu)
        db.load_scenes(con, scenes)
        db.load_station_values(con, extracts)
        db.load_matchups(con, m)
        tbl = db.matchup_spectra(con, "chl")

        # ---- OWT on all spectra
        scheme = owt.OWTScheme(k_min=3, k_max=5).fit(tbl).fit_thresholds(tbl)
        tbl = tbl.drop(columns=["owt"]).join(scheme.predict(tbl))
        con.executemany("UPDATE matchup SET owt=? WHERE matchup_id=?", [(int(o), int(i)) for o, i in zip(tbl["owt"], tbl["matchup_id"])])

        # ---- baselines on Tokyo Bay: train 2024–25, test 2026
        tk = tbl[tbl["bay"] == "tokyo"]
        tr, te = tk[tk["scene_time"].dt.year < 2026], tk[tk["scene_time"].dt.year == 2026]
        tiers = baseline.run_tiers(tr, te, "chl")
        tiers.to_csv(out / "baseline_tiers.csv", index=False)
        for r in tiers.itertuples(index=False):
            run = db.new_run(con, r.algorithm, "chl", int(r.tier), "tokyo 2024-2025")
            db.save_metrics(con, run, "split:test_time", r._asdict())

        # ---- tier-3 model with uncertainty
        rf = models.RFRetrieval().fit(tr, tr["y"])
        pred = rf.predict(te)
        run = db.new_run(con, "rf_owt", "chl", 3, "tokyo 2024-2025", {"features": rf.feature_names[:5]})
        db.save_predictions(con, run, te["matchup_id"], pred["y_pred"], pred["y_lo"], pred["y_hi"])
        m3 = baseline.metrics(te["y"], pred["y_pred"], pred["y_lo"], pred["y_hi"])
        db.save_metrics(con, run, "split:test_time", m3)
        models.calibration_table(te["y"], pred).to_csv(out / "calibration.csv", index=False)

        # ---- transfer experiments
        transfer.temporal_holdout(tk, [2024, 2025], 2026).to_csv(out / "temporal_holdout.csv", index=False)
        transfer.static_vs_adaptive(tk, [2024, 2025], 2026).to_csv(out / "static_vs_adaptive.csv", index=False)
        sp = transfer.spatial_transfer(tbl, "tokyo", "sagami", scheme)
        pd.DataFrame([{"experiment": sp["experiment"], "share_novel_owt": sp["share_novel"], **{f"all_{k}": v for k, v in sp["metrics_all"].items()},
                       **{f"seen_{k}": v for k, v in sp["metrics_seen_owt"].items()}, **{f"novel_{k}": v for k, v in sp["metrics_novel_owt"].items()}}]
                     ).to_csv(out / "spatial_transfer.csv", index=False)
        sp["coverage"].to_csv(out / "owt_coverage.csv")
        transfer.learning_curve(tbl, "tokyo", "sagami", sizes=(0, 5, 10, 20, 40), repeats=3).to_csv(out / "learning_curve.csv")
        scheme.describe().to_csv(out / "owt_centres.csv")

    # ---- RQ3 lag analysis on in-situ
    daily = lag.daily_series(insitu)
    lag.lagged_correlation(daily).to_csv(out / "lag_correlation.csv", index=False)
    ev = lag.hypoxia_events(daily)
    ev.to_csv(out / "hypoxia_events.csv", index=False)
    lag.event_lead_time(daily, ev).to_csv(out / "event_lead_time.csv", index=False)

    # ---- Power BI export
    files = export.export(dbp, out)

    summary = {"db": dbp, "n_insitu": len(insitu), "n_scenes": len(scenes), "n_matchups": len(m), "owt_k": scheme.k,
               "tier3_metrics": m3, "n_hypoxia_events": len(ev), "powerbi_files": len(files)}
    log.info("demo built: %s", summary)
    return summary
