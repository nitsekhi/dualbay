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
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows → {out}")


def cmd_study_area(a):
    zs = study_area.zones()
    if a.coastline:
        zs = study_area.clip_to_water(zs, study_area.load_land(a.coastline, config.load("study_area")["bbox"]))
    print(f"Saved {len(zs)} zones → {study_area.to_geojson(zs, a.out)}")


def cmd_parse_insitu(a):
    if a.station:
        df = pd.concat([insitu.parse_file(f, station=a.station) for f in a.files], ignore_index=True)
    else:
        df = insitu.parse_many(a.files)
    _save(df, a.out)
    print(df.groupby(["station", "param"])["value"].agg(["count", "min", "max"]).to_string())


def cmd_inventory(a):
    _save(inventory.build(a.start, a.end, a.sensors), a.out)


def cmd_matchups(a):
    obs = pd.read_csv(a.insitu, parse_dates=["time"])
    obs["time"] = pd.to_datetime(obs["time"], utc=True).dt.tz_convert(insitu.TZ)
    scenes = pd.read_csv(a.scenes)
    m = matchup.find_matchups(obs, scenes, window_h=a.window)
    _save(m, a.out)
    s = matchup.summary(m)
    _save(s, Path(a.out).with_name("matchup_summary.csv"))
    print(s.to_string(index=False))


def cmd_pace(a):
    from . import pace

    if a.files:
        pace.process_files(a.files, a.out)
    else:
        pace.process(a.start, a.end, pace.AOP if a.product == "aop" else pace.BGC, a.max_cloud, a.limit, a.out)


def cmd_figures(a):
    from . import figures

    for p in figures.make_all(a.pace_dir, a.out):
        print(f"Saved {p}")


def main(argv=None):
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
    s.add_argument("--product", choices=["aop", "bgc"], default="aop", help="aop = Rrs spectra, bgc = chlor_a")
    s.add_argument("--max-cloud", type=float, default=80, help="granule-wide cloud cover limit, %%")
    s.add_argument("--limit", type=int, help="max number of granules (for a quick test)")
    s.add_argument("--files", nargs="*", help="local .nc files instead of a remote search")
    s.add_argument("--out", default=PROCESSED / "pace")
    s.set_defaults(func=cmd_pace)

    s = sub.add_parser("figures", help="First-look figures from the PACE tables")
    s.add_argument("--pace-dir", default=PROCESSED / "pace")
    s.add_argument("--out", default="figures")
    s.set_defaults(func=cmd_figures)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
