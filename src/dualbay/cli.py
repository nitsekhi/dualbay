"""Command line: `dualbay <command>`.

dualbay study-area   [--coastline land.geojson] [--out data/processed/study_area.geojson]
dualbay parse-insitu FILE [FILE ...] [--station ID] [--out data/processed/insitu.csv]
dualbay inventory    --start 2024-03-01 --end 2026-09-30 [--sensors s3_olci pace_oci] [--out ...]
dualbay matchups     --insitu data/processed/insitu.csv --scenes data/processed/scenes.csv [--out ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config, insitu, inventory, matchup, study_area

PROCESSED = Path("data/processed")


def _save(df: pd.DataFrame, out: str | Path) -> None:
    """Write a table to CSV, creating the folder if needed, and report the row count."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows → {out}")


def cmd_study_area(a):
    """`dualbay study-area`: write the five zones as GeoJSON (optionally clipped to water)."""
    zs = study_area.zones()
    if a.coastline:
        zs = study_area.clip_to_water(
            zs, study_area.load_land(a.coastline, config.load("study_area")["bbox"])
        )
    print(f"Saved {len(zs)} zones → {study_area.to_geojson(zs, a.out)}")


def cmd_parse_insitu(a):
    """`dualbay parse-insitu`: parse downloaded station files into one long table."""
    if a.station:
        df = pd.concat([insitu.parse_file(f, station=a.station) for f in a.files], ignore_index=True)
    else:
        df = insitu.parse_many(a.files)
    _save(df, a.out)
    print(df.groupby(["station", "param"])["value"].agg(["count", "min", "max"]).to_string())


def cmd_inventory(a):
    """`dualbay inventory`: list satellite scenes over the study area for a date range."""
    _save(inventory.build(a.start, a.end, a.sensors), a.out)


def cmd_matchups(a):
    """`dualbay matchups`: pair scenes with station observations and print the summary table."""
    obs = pd.read_csv(a.insitu, parse_dates=["time"])
    obs["time"] = pd.to_datetime(obs["time"], utc=True).dt.tz_convert(insitu.TZ)
    scenes = pd.read_csv(a.scenes)
    m = matchup.find_matchups(obs, scenes, window_h=a.window)
    _save(m, a.out)
    s = matchup.summary(m)
    _save(s, Path(a.out).with_name("matchup_summary.csv"))
    print(s.to_string(index=False))


def cmd_pace(a):
    """`dualbay pace`: read PACE granules (searched or given as files) and write the zone/station tables."""
    from . import pace

    if a.files:
        pace.process_files(a.files, a.out)
    else:
        pace.process(
            a.start, a.end, pace.AOP if a.product == "aop" else pace.BGC, a.max_cloud, a.limit, a.out
        )


def cmd_figures(a):
    """`dualbay figures`: draw the summary figures from the PACE tables."""
    from . import figures

    for p in figures.make_all(a.pace_dir, a.out):
        print(f"Saved {p}")


def cmd_load(a):
    """Load processed CSVs into the SQLite store."""
    from . import db

    with db.connect(a.db) as con:
        db.load_dimensions(con)
        if a.insitu:
            df = pd.read_csv(a.insitu, parse_dates=["time"])
            print(f"insitu: {db.load_insitu(con, df)} new rows")
        if a.scenes:
            print(f"scenes: {db.load_scenes(con, pd.read_csv(a.scenes))} new rows")
        if a.zone_spectra:
            print(f"zone stats: {db.load_zone_stats(con, pd.read_csv(a.zone_spectra))} rows")
        if a.station_values:
            print(f"pixel extracts: {db.load_station_values(con, pd.read_csv(a.station_values))} rows")
        if a.matchups:
            print(f"matchups: {db.load_matchups(con, pd.read_csv(a.matchups))} new rows")


def cmd_analyze(a):
    """Baselines (tiers 1–2), tier-3 model with uncertainty, transfer experiments; results to CSV + store."""
    from . import baseline, db, models, owt, transfer

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with db.connect(a.db) as con:
        tbl = db.matchup_spectra(con, a.param)
        if len(tbl) < 40:
            raise SystemExit(f"only {len(tbl)} matchups with spectra for {a.param}; need ≥ 40")
        scheme = owt.OWTScheme().fit(tbl).fit_thresholds(tbl)
        tbl = tbl.drop(columns=["owt"]).join(scheme.predict(tbl))
        con.executemany(
            "UPDATE matchup SET owt=? WHERE matchup_id=?",
            [(int(o), int(i)) for o, i in zip(tbl["owt"], tbl["matchup_id"])],
        )
        scheme.describe().to_csv(out / "owt_centres.csv")
        src = tbl[tbl["bay"] == a.train_bay]
        years = sorted(src["scene_time"].dt.year.unique())
        if len(years) < 2:
            raise SystemExit("need at least two years of matchups for a temporal holdout")
        train_years, test_year = years[:-1], years[-1]
        tr, te = src[src["scene_time"].dt.year.isin(train_years)], src[src["scene_time"].dt.year == test_year]
        tiers = baseline.run_tiers(tr, te, a.param)
        tiers.to_csv(out / "baseline_tiers.csv", index=False)
        for r in tiers.itertuples(index=False):
            run = db.new_run(con, r.algorithm, a.param, int(r.tier), f"{a.train_bay} {train_years}")
            db.save_metrics(con, run, f"split:test_{test_year}", r._asdict())
        rf = models.RFRetrieval().fit(tr, tr["y"])
        pred = rf.predict(te)
        run = db.new_run(con, "rf_owt", a.param, 3, f"{a.train_bay} {train_years}")
        db.save_predictions(con, run, te["matchup_id"], pred["y_pred"], pred["y_lo"], pred["y_hi"])
        db.save_metrics(
            con,
            run,
            f"split:test_{test_year}",
            baseline.metrics(te["y"], pred["y_pred"], pred["y_lo"], pred["y_hi"]),
        )
        rf.importance().to_csv(out / "feature_importance.csv")
        models.calibration_table(te["y"], pred).to_csv(out / "calibration.csv", index=False)
        transfer.temporal_holdout(src, train_years, test_year).to_csv(
            out / "temporal_holdout.csv", index=False
        )
        transfer.static_vs_adaptive(src, train_years, test_year).to_csv(
            out / "static_vs_adaptive.csv", index=False
        )
        other = [b for b in tbl["bay"].unique() if b != a.train_bay]
        for b in other:
            try:
                sp = transfer.spatial_transfer(tbl, a.train_bay, b, scheme)
                sp["coverage"].to_csv(out / f"owt_coverage_{a.train_bay}_to_{b}.csv")
                pd.DataFrame(
                    [
                        {
                            "experiment": sp["experiment"],
                            "share_novel_owt": sp["share_novel"],
                            **sp["metrics_all"],
                        }
                    ]
                ).to_csv(out / f"spatial_transfer_{a.train_bay}_to_{b}.csv", index=False)
                transfer.learning_curve(tbl, a.train_bay, b).to_csv(
                    out / f"learning_curve_{a.train_bay}_to_{b}.csv"
                )
            except ValueError as err:
                print(f"spatial transfer {a.train_bay} → {b} skipped: {err}")
    print(f"Results in {out}")


def cmd_lag(a):
    """`dualbay lag`: surface chlorophyll vs bottom oxygen lags, hypoxia events and lead times."""
    from . import lag

    ins = pd.read_csv(a.insitu, parse_dates=["time"])
    daily = lag.daily_series(ins)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out / "daily_series.csv", index=False)
    lag.lagged_correlation(daily).to_csv(out / "lag_correlation.csv", index=False)
    ev = lag.hypoxia_events(daily, a.threshold)
    ev.to_csv(out / "hypoxia_events.csv", index=False)
    lag.event_lead_time(daily, ev).to_csv(out / "event_lead_time.csv", index=False)
    print(f"{len(ev)} hypoxia events (< {a.threshold} mg/L); results in {out}")


def cmd_export(a):
    """`dualbay export`: write the Power BI tables and the DAX measures file."""
    from . import export

    for p in export.export(a.db, a.out):
        print(f"Saved {p}")


def cmd_demo(a):
    """`dualbay demo`: run the whole chain on synthetic data."""
    import logging

    from . import demo

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    s = demo.build(a.out)
    print({k: v for k, v in s.items() if k != "tier3_metrics"})
    print("tier-3 metrics:", {k: round(v, 3) for k, v in s["tier3_metrics"].items()})


def main(argv=None):
    """Parse the command line and dispatch to the `cmd_*` function for the chosen sub-command."""
    p = argparse.ArgumentParser(prog="dualbay", description="Tokyo Bay + Sagami Bay water quality toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("study-area", help="Write study-area zones as GeoJSON")
    s.add_argument("--coastline", help="GeoJSON with land polygons to clip zones to water")
    s.add_argument("--out", default=PROCESSED / "study_area.geojson")
    s.set_defaults(func=cmd_study_area)

    s = sub.add_parser("parse-insitu", help="Convert downloaded in-situ CSV/Excel files to one long table")
    s.add_argument("files", nargs="+")
    s.add_argument("--station", help="Station id for files without a station column (default: file name)")
    s.add_argument("--out", default=PROCESSED / "insitu.csv")
    s.set_defaults(func=cmd_parse_insitu)

    s = sub.add_parser("inventory", help="List satellite scenes over the study area")
    s.add_argument("--start", required=True)
    s.add_argument("--end", required=True)
    s.add_argument("--sensors", nargs="*")
    s.add_argument("--out", default=PROCESSED / "scenes.csv")
    s.set_defaults(func=cmd_inventory)

    s = sub.add_parser("matchups", help="Pair scenes with in-situ observations and count matchups")
    s.add_argument("--insitu", required=True)
    s.add_argument("--scenes", required=True)
    s.add_argument("--window", type=float, help="± hours (default from sensors.yaml)")
    s.add_argument("--out", default=PROCESSED / "matchups.csv")
    s.set_defaults(func=cmd_matchups)

    s = sub.add_parser("pace", help="PACE OCI L2: zone spectra and station values over the bays")
    s.add_argument("--start", default="2024-04-01")
    s.add_argument("--end", default="2026-09-30")
    s.add_argument(
        "--product", choices=["aop", "bgc"], default="aop", help="aop = Rrs spectra, bgc = chlor_a"
    )
    s.add_argument("--max-cloud", type=float, default=80, help="granule-wide cloud cover limit, %%")
    s.add_argument("--limit", type=int, help="max number of granules (for a quick test)")
    s.add_argument("--files", nargs="*", help="local .nc files instead of a remote search")
    s.add_argument("--out", default=PROCESSED / "pace")
    s.set_defaults(func=cmd_pace)

    s = sub.add_parser("figures", help="First-look figures from the PACE tables")
    s.add_argument("--pace-dir", default=PROCESSED / "pace")
    s.add_argument("--out", default="figures")
    s.set_defaults(func=cmd_figures)

    s = sub.add_parser("load", help="Load processed CSVs into the SQLite store")
    s.add_argument("--db", default="data/dualbay.sqlite")
    s.add_argument("--insitu")
    s.add_argument("--scenes")
    s.add_argument("--zone-spectra", dest="zone_spectra")
    s.add_argument("--station-values", dest="station_values")
    s.add_argument("--matchups")
    s.set_defaults(func=cmd_load)

    s = sub.add_parser("analyze", help="Baselines, uncertainty model and transfer experiments from the store")
    s.add_argument("--db", default="data/dualbay.sqlite")
    s.add_argument("--param", default="chl")
    s.add_argument("--train-bay", dest="train_bay", default="tokyo")
    s.add_argument("--out", default="results")
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("lag", help="RQ3 groundwork: surface Chl-a vs bottom DO lags and hypoxia events")
    s.add_argument("--insitu", required=True)
    s.add_argument("--threshold", type=float, default=3.0)
    s.add_argument("--out", default="results/lag")
    s.set_defaults(func=cmd_lag)

    s = sub.add_parser("export", help="Power BI star schema + DAX measures")
    s.add_argument("--db", default="data/dualbay.sqlite")
    s.add_argument("--out", default=PROCESSED)
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("demo", help="Offline end-to-end run on synthetic data (data/demo)")
    s.add_argument("--out", default="data/demo")
    s.set_defaults(func=cmd_demo)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
