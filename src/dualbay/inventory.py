"""Scene inventory: which satellite scenes cover the study area, and when.

Output columns: sensor | scene_id | time (UTC) | cloud_pct | lon_min | lat_min | lon_max | lat_max

Searchable providers:
  cdse       Copernicus Data Space STAC (Sentinel-2, Sentinel-3)  — needs `pip install pystac-client`
  earthdata  NASA CMR via earthaccess (PACE)                       — needs `pip install earthaccess` and an Earthdata login
Portals without a search API (G-Portal, PRISMA, Tellus): export the scene list from the portal
as CSV and load it with `load_manual_list`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config

CDSE_STAC = "https://stac.dataspace.copernicus.eu/v1"
COLUMNS = ["sensor", "scene_id", "time", "cloud_pct", "lon_min", "lat_min", "lon_max", "lat_max"]


def search_cdse(sensor_id: str, spec: dict, bbox, start: str, end: str, max_items: int = 5000) -> pd.DataFrame:
    from pystac_client import Client  # optional dependency

    client = Client.open(CDSE_STAC)
    query = {}
    if "max_cloud_pct" in spec:
        query["eo:cloud_cover"] = {"lte": spec["max_cloud_pct"]}
    search = client.search(
        collections=[spec["collection"]], bbox=bbox, datetime=f"{start}/{end}", max_items=max_items, query=query or None
    )
    rows = []
    for item in search.items():
        b = item.bbox or [None] * 4
        rows.append(
            {
                "sensor": sensor_id,
                "scene_id": item.id,
                "time": item.datetime,
                "cloud_pct": item.properties.get("eo:cloud_cover"),
                "lon_min": b[0], "lat_min": b[1], "lon_max": b[2], "lat_max": b[3],
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def search_earthdata(sensor_id: str, spec: dict, bbox, start: str, end: str) -> pd.DataFrame:
    import earthaccess  # optional dependency

    earthaccess.login(strategy="netrc")
    results = earthaccess.search_data(short_name=spec["short_name"], bounding_box=tuple(bbox), temporal=(start, end))
    rows = []
    for g in results:
        umm = g["umm"]
        t = umm["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
        rect = (
            umm.get("SpatialExtent", {}).get("HorizontalSpatialDomain", {}).get("Geometry", {}).get("BoundingRectangles")
            or [{}]
        )[0]
        rows.append(
            {
                "sensor": sensor_id,
                "scene_id": umm.get("GranuleUR"),
                "time": t,
                "cloud_pct": umm.get("CloudCover"),
                "lon_min": rect.get("WestBoundingCoordinate"), "lat_min": rect.get("SouthBoundingCoordinate"),
                "lon_max": rect.get("EastBoundingCoordinate"), "lat_max": rect.get("NorthBoundingCoordinate"),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def load_manual_list(path: str | Path, sensor_id: str, time_col: str = "time", id_col: str = "scene_id") -> pd.DataFrame:
    """Load a scene list exported from a portal (G-Portal, PRISMA, Tellus). Needs at least a time and an id column."""
    df = pd.read_csv(path)
    out = pd.DataFrame({"sensor": sensor_id, "scene_id": df[id_col], "time": df[time_col]})
    for c in COLUMNS[3:]:
        out[c] = df[c] if c in df.columns else None
    return out[COLUMNS]


def build(start: str, end: str, sensors: list[str] | None = None) -> pd.DataFrame:
    """Search every configured sensor with a search API over the study-area bbox."""
    area = config.load("study_area")
    cfg = config.load("sensors")["sensors"]
    frames = []
    for sid, spec in cfg.items():
        if sensors and sid not in sensors:
            continue
        if spec["provider"] == "cdse":
            frames.append(search_cdse(sid, spec, area["bbox"], start, end))
        elif spec["provider"] == "earthdata":
            frames.append(search_earthdata(sid, spec, area["bbox"], start, end))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values(["sensor", "time"]).reset_index(drop=True)
