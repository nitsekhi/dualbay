"""Study area zones: build them, clip them to water, and look up which zone a point is in."""

from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import Point, box, mapping, shape
from shapely.ops import unary_union

from . import config


def zones(cfg: dict | None = None) -> dict[str, dict]:
    """Return {zone_id: {"geometry": shapely geometry, **attributes}} from study_area.yaml."""
    cfg = cfg or config.load("study_area")
    out = {}
    for zid, z in cfg["zones"].items():
        out[zid] = {"geometry": box(*z["bbox"]), "bay": z["bay"], "description": z["description"]}
    return out


def load_land(path: str | Path, bbox: list[float] | None = None):
    """Read land polygons from a GeoJSON file (e.g. OSM land polygons or GSHHG, clipped to the region)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    feats = data["features"] if data.get("type") == "FeatureCollection" else [data]
    land = unary_union([shape(f["geometry"]) for f in feats])
    return land.intersection(box(*bbox)) if bbox else land


def clip_to_water(zs: dict[str, dict], land) -> dict[str, dict]:
    """Subtract land from every zone."""
    return {zid: {**z, "geometry": z["geometry"].difference(land)} for zid, z in zs.items()}


def to_geojson(zs: dict[str, dict], path: str | Path) -> Path:
    """Write the zones as a GeoJSON FeatureCollection and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"zone": zid, "bay": z["bay"], "description": z["description"]},
                "geometry": mapping(z["geometry"]),
            }
            for zid, z in zs.items()
        ],
    }
    path.write_text(json.dumps(fc, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def zone_of(lon: float, lat: float, zs: dict[str, dict] | None = None) -> str | None:
    """Zone id containing the point, or None. Overlapping zones resolve to the first listed."""
    zs = zs or zones()
    p = Point(lon, lat)
    for zid, z in zs.items():
        if z["geometry"].covers(p):
            return zid
    return None
