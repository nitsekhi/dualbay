"""PACE OCI Level-2: find granules over the bays, read only the bay subset, extract zone and station spectra.

Products
  PACE_OCI_L2_AOP   remote sensing reflectance Rrs(λ), ~184 bands 340–890 nm
  PACE_OCI_L2_BGC   standard biogeochemical products, incl. chlor_a

Granules are large (hundreds of MB). `process` opens them remotely through earthaccess and
reads only the rows/columns that cover the study area, so nothing big is downloaded.
Local .nc files work too (`process_files`).

Output (long tables, CSV):
  zone_spectra.csv     time | granule | zone | bay | n_water | n_valid | valid_frac | wavelength | rrs_median | rrs_p25 | rrs_p75 | rrs_unc_median
  station_values.csv   time | granule | station | dist_km | n_valid | passed | wavelength | rrs | chlor_a
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, flags, matchup, study_area

log = logging.getLogger(__name__)
RETRIES = 3
RETRY_WAIT_S = 20

AOP = "PACE_OCI_L2_AOP"
BGC = "PACE_OCI_L2_BGC"
LAND_BIT = 1 << flags.L2_FLAG_BITS["LAND"]
REF_WL = 555.0  # band used to decide whether a pixel is valid
BLUE_WL = (412.0, 443.0)  # bands checked for negative Rrs (atmospheric over-correction, Higa et al. 2024)
MAX_STATION_DIST_KM = 1.5  # nearest pixel must be within this distance (OCI pixel ~1.2 km)
MASK_LEVEL = "standard"  # flags.STANDARD_MASK; set "water" to keep more (riskier) pixels
MASK_EXTRA = ()  # e.g. ("ABSAER", "ATMWARN") for a conservative run
ACCESS_MODE = (
    "download"  # "download": fetch granule to a temp dir, subset, delete (reliable outside AWS us-west-2)
)
# "open": stream remotely (fast inside us-west-2, flaky elsewhere)


# ---------- search ----------


def login():
    """Log in to NASA Earthdata from ~/.netrc, falling back to an interactive prompt."""
    import earthaccess

    return earthaccess.login(strategy="netrc") or earthaccess.login(strategy="interactive")


def search(
    start: str, end: str, short_name: str = AOP, max_cloud: float | None = None, limit: int | None = None
):
    """earthaccess search over the study-area bbox. `max_cloud` filters on the granule-wide cloud cover."""
    import earthaccess

    kw = dict(
        short_name=short_name, bounding_box=tuple(config.load("study_area")["bbox"]), temporal=(start, end)
    )
    if max_cloud is not None:
        kw["cloud_cover"] = (0, max_cloud)
    if limit:
        kw["count"] = limit
    return earthaccess.search_data(**kw)


def granule_table(results) -> pd.DataFrame:
    """Flatten Earthdata search results into a table (granule name, time, cloud cover, bounding box, size)."""
    rows = []
    for g in results:
        umm = g["umm"]
        rows.append(
            {
                "granule": umm.get("GranuleUR"),
                "time": umm["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"],
                "cloud_pct": umm.get("CloudCover"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


# ---------- reading ----------


def _read(ds, sl=None) -> np.ndarray:
    """Read an HDF5 dataset applying _FillValue, scale_factor and add_offset."""
    a = ds[sl] if sl is not None else ds[...]
    a = np.asarray(a)
    fill = ds.attrs.get("_FillValue")
    out = a.astype("float64")
    if fill is not None:
        out[a == np.asarray(fill).item()] = np.nan
    scale = ds.attrs.get("scale_factor")
    offset = ds.attrs.get("add_offset")
    if scale is not None:
        out = out * float(np.asarray(scale).item())
    if offset is not None:
        out = out + float(np.asarray(offset).item())
    return out


def read_subset(src, bbox, variables=("Rrs",)) -> dict | None:
    """Read the bbox subset of a PACE L2 file (path or file-like object). None if the swath misses the bbox.

    Returns dict with lat, lon (2-D), flags (2-D or None), wavelength (1-D or None) and requested variables.
    """
    import h5py

    lon0, lat0, lon1, lat1 = bbox
    with h5py.File(src, "r") as f:
        lat = _read(f["navigation_data/latitude"])
        lon = _read(f["navigation_data/longitude"])
        inside = (lat >= lat0) & (lat <= lat1) & (lon >= lon0) & (lon <= lon1)
        if not inside.any():
            return None
        rows, cols = np.where(inside.any(axis=1))[0], np.where(inside.any(axis=0))[0]
        r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        out = {
            "lat": lat[r0:r1, c0:c1],
            "lon": lon[r0:r1, c0:c1],
            "flags": None,
            "wavelength": None,
            "flag_table": None,
            "product_version": None,
        }
        pv = f.attrs.get("processing_version") or f.attrs.get("product_version")
        if pv is not None:
            out["product_version"] = pv.decode() if isinstance(pv, bytes) else str(pv)
        geo = f["geophysical_data"]
        if "l2_flags" in geo:
            out["flags"] = np.asarray(geo["l2_flags"][r0:r1, c0:c1])
            fm = geo["l2_flags"].attrs.get("flag_meanings")
            fm = fm.decode() if isinstance(fm, bytes) else fm
            out["flag_table"] = flags.from_file_attrs(fm, geo["l2_flags"].attrs.get("flag_masks"))
        if "sensor_band_parameters/wavelength_3d" in f:
            out["wavelength"] = np.asarray(f["sensor_band_parameters/wavelength_3d"][...], dtype=float)
        for v in list(variables) + (["Rrs_unc"] if "Rrs" in variables else []):
            if v in geo:
                ds = geo[v]
                out[v] = _read(ds, (slice(r0, r1), slice(c0, c1)) + ((slice(None),) if ds.ndim == 3 else ()))
        return out


def granule_time(name: str) -> pd.Timestamp | None:
    """PACE_OCI.20250714T031234.L2... → 2025-07-14 03:12:34 UTC"""
    try:
        return pd.to_datetime(Path(str(name)).name.split(".")[1], format="%Y%m%dT%H%M%S", utc=True)
    except (IndexError, ValueError):
        return None


# ---------- extraction ----------


def _water_valid(sub: dict, ref_idx: int | None):
    """water = not land; valid = water and passes the NASA standard mask and has a finite value."""
    shape = sub["lat"].shape
    fl = sub["flags"]
    tb = sub.get("flag_table")
    water = np.ones(shape, bool) if fl is None else ~flags.is_set(fl, "LAND", tb)
    good = flags.good_pixels(fl, shape, MASK_LEVEL, tb, MASK_EXTRA)
    if "Rrs" in sub and ref_idx is not None:
        valid = water & good & np.isfinite(sub["Rrs"][..., ref_idx])
    elif "chlor_a" in sub:
        valid = water & good & np.isfinite(sub["chlor_a"])
    else:
        valid = water & good
    return water, valid


def pixel_budget(sub: dict, zones: dict | None = None) -> pd.DataFrame:
    """How many usable pixels each zone really has, and why pixels are lost.

    Columns: zone, n_water, n_good (standard mask), frac_good, frac_<flag> for the diagnostic flags,
    and, when Rrs is present, frac_neg_412 / frac_neg_443 among good pixels.
    Closes the 'PACE resolution and adjacency' gap: inner Tokyo Bay is ~20 km wide, so a 1.2 km
    pixel grid leaves few pixels free of land adjacency and stray light.
    """
    zones = zones or study_area.zones()
    wl = sub["wavelength"]
    fl = sub["flags"]
    tb = sub.get("flag_table")
    shape = sub["lat"].shape
    water = np.ones(shape, bool) if fl is None else ~flags.is_set(fl, "LAND", tb)
    good = water & flags.good_pixels(fl, shape, MASK_LEVEL, tb, MASK_EXTRA)
    rows = []
    for zid, z in zones.items():
        x0, y0, x1, y1 = z["geometry"].bounds
        inz = (sub["lon"] >= x0) & (sub["lon"] <= x1) & (sub["lat"] >= y0) & (sub["lat"] <= y1)
        n_water, n_good = int((inz & water).sum()), int((inz & good).sum())
        row = {
            "zone": zid,
            "bay": z["bay"],
            "n_water": n_water,
            "n_good": n_good,
            "frac_good": n_good / n_water if n_water else np.nan,
        }
        row.update(flags.flag_fractions(fl, inz & water, table=tb) if fl is not None else {})
        if "Rrs" in sub and wl is not None and n_good:
            for b in BLUE_WL:
                k = int(np.nanargmin(np.abs(wl - b)))
                v = sub["Rrs"][..., k][inz & good]
                row[f"frac_neg_{int(b)}"] = float(np.mean(v < 0)) if v.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def zone_spectra(sub: dict, granule: str, zones: dict | None = None) -> pd.DataFrame:
    """Median/IQR Rrs spectrum of valid water pixels per zone, plus how much of the zone was clear."""
    zones = zones or study_area.zones()
    wl = sub["wavelength"]
    ref_idx = int(np.nanargmin(np.abs(wl - REF_WL))) if wl is not None else None
    water, valid = _water_valid(sub, ref_idx)
    t = granule_time(granule)
    budget = pixel_budget(sub, zones).set_index("zone")
    rows = []
    for zid, z in zones.items():
        x0, y0, x1, y1 = z["geometry"].bounds
        inz = (sub["lon"] >= x0) & (sub["lon"] <= x1) & (sub["lat"] >= y0) & (sub["lat"] <= y1)
        n_water, n_valid = int((inz & water).sum()), int((inz & valid).sum())
        diag = {c: budget.loc[zid, c] for c in budget.columns if c.startswith("frac_")}
        base = {
            "time": t,
            "granule": granule,
            "product_version": sub.get("product_version"),
            "zone": zid,
            "bay": z["bay"],
            "n_water": n_water,
            "n_valid": n_valid,
            "valid_frac": n_valid / n_water if n_water else 0.0,
            **diag,
        }
        if n_valid == 0 or "Rrs" not in sub:
            rows.append(
                {**base, "wavelength": np.nan, "rrs_median": np.nan, "rrs_p25": np.nan, "rrs_p75": np.nan}
            )
            continue
        spec = sub["Rrs"][inz & valid]  # (n_pixels, n_bands)
        q25, med, q75 = np.nanpercentile(spec, [25, 50, 75], axis=0)
        unc = (
            np.nanmedian(sub["Rrs_unc"][inz & valid], axis=0)
            if "Rrs_unc" in sub
            else np.full(len(wl), np.nan)
        )
        for k, w in enumerate(wl):
            rows.append(
                {
                    **base,
                    "wavelength": float(w),
                    "rrs_median": med[k],
                    "rrs_p25": q25[k],
                    "rrs_p75": q75[k],
                    "rrs_unc_median": unc[k],
                }
            )
    return pd.DataFrame(rows)


def _dist_km(lat, lon, lat0, lon0):
    """Approximate distance in km between a pixel and a station (equirectangular)."""
    k = np.cos(np.radians(lat0))
    return 111.2 * np.hypot(lat - lat0, (lon - lon0) * k)


def station_values(sub: dict, granule: str, stations: pd.DataFrame | None = None) -> pd.DataFrame:
    """3×3 window around the nearest pixel to each station (Bailey & Werdell 2006 criteria on the 555 nm band
    or on chlor_a). Returns the window median spectrum and/or chlor_a."""
    stations = stations if stations is not None else matchup.stations_frame()
    wl = sub["wavelength"]
    ref_idx = int(np.nanargmin(np.abs(wl - REF_WL))) if (wl is not None and "Rrs" in sub) else None
    water, _ = _water_valid(sub, ref_idx)
    t = granule_time(granule)
    rows = []
    for s in stations.itertuples(index=False):
        d = _dist_km(sub["lat"], sub["lon"], s.lat, s.lon)
        r, c = np.unravel_index(np.nanargmin(d), d.shape)
        dist = float(d[r, c])
        if dist > MAX_STATION_DIST_KM:
            continue
        ref = sub["Rrs"][..., ref_idx] if ref_idx is not None else sub.get("chlor_a")
        if ref is None:
            continue
        ref = np.where(water, ref, np.nan)
        ws = matchup.window_stats(ref, r, c)
        base = {
            "time": t,
            "granule": granule,
            "station": s.station,
            "dist_km": round(dist, 2),
            "n_valid": ws["n_valid"],
            "passed": ws["passed"],
        }
        r0, r1, c0, c1 = max(r - 1, 0), r + 2, max(c - 1, 0), c + 2
        chl = float(np.nanmedian(sub["chlor_a"][r0:r1, c0:c1])) if "chlor_a" in sub else np.nan
        if ref_idx is not None:
            spec = np.nanmedian(sub["Rrs"][r0:r1, c0:c1, :].reshape(-1, len(wl)), axis=0)
            unc = (
                np.nanmedian(sub["Rrs_unc"][r0:r1, c0:c1, :].reshape(-1, len(wl)), axis=0)
                if "Rrs_unc" in sub
                else np.full(len(wl), np.nan)
            )
            for k, w in enumerate(wl):
                rows.append(
                    {
                        **base,
                        "wavelength": float(w),
                        "rrs": spec[k],
                        "rrs_unc": unc[k],
                        "chlor_a": chl,
                        "product_version": sub.get("product_version"),
                    }
                )
        else:
            rows.append(
                {
                    **base,
                    "wavelength": np.nan,
                    "rrs": np.nan,
                    "rrs_unc": np.nan,
                    "chlor_a": chl,
                    "product_version": sub.get("product_version"),
                }
            )
    return pd.DataFrame(rows)


# ---------- drivers ----------


def _process_one(src, name, bbox, variables, out_z, out_s):
    """Read one granule's bay subset and append its zone and station tables; False if the granule has no bay pixels."""
    sub = read_subset(src, bbox, variables)
    if sub is None:
        return False
    out_z.append(zone_spectra(sub, name))
    out_s.append(station_values(sub, name))
    return True


def _save(frames, path: Path) -> pd.DataFrame:
    """Concatenate the per-granule tables and write them to CSV."""
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def process_files(
    paths, out_dir="data/processed/pace", variables=("Rrs", "chlor_a")
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Local PACE L2 files (AOP and/or BGC)."""
    bbox = config.load("study_area")["bbox"]
    zs, ss = [], []
    for p in paths:
        ok = _process_one(p, Path(p).name, bbox, variables, zs, ss)
        print(("✓ " if ok else "– outside bbox ") + Path(p).name)
    out = Path(out_dir)
    return _save(zs, out / "zone_spectra.csv"), _save(ss, out / "station_values.csv")


def process(
    start: str,
    end: str,
    short_name: str = AOP,
    max_cloud: float | None = 80,
    limit: int | None = None,
    out_dir="data/processed/pace",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Search, open each granule remotely, read the bay subset, write the tables. Saves progress every 10 granules."""
    import earthaccess

    login()
    results = search(start, end, short_name, max_cloud, limit)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if short_name == AOP else "_bgc"
    granule_table(results).to_csv(out / f"granules{suffix}.csv", index=False)
    print(f"{len(results)} granules found")
    bbox = config.load("study_area")["bbox"]
    variables = ("Rrs",) if short_name == AOP else ("chlor_a",)
    zs, ss = [], []
    done = set()
    resume = out / f"zone_spectra{suffix}.csv"
    if resume.exists():  # resume an interrupted run
        try:
            prev = pd.read_csv(resume)
            done = set(prev["granule"].unique())
            zs.append(prev)
            if (out / f"station_values{suffix}.csv").exists():
                ss.append(pd.read_csv(out / f"station_values{suffix}.csv"))
            log.info("resuming: %d granules already processed", len(done))
        except Exception as err:
            log.warning("could not read previous output, starting fresh: %s", err)
    failures = []
    for i, g in enumerate(results, 1):
        name = g["umm"]["GranuleUR"]
        if name in done:
            continue
        for attempt in range(1, RETRIES + 1):
            try:
                if ACCESS_MODE == "download":
                    import tempfile

                    with tempfile.TemporaryDirectory(prefix="pace_") as td:
                        local = earthaccess.download([g], td)[0]
                        ok = _process_one(local, name, bbox, variables, zs, ss)
                else:
                    fobj = earthaccess.open([g])[0]
                    ok = _process_one(fobj, name, bbox, variables, zs, ss)
                print(f"[{i}/{len(results)}] {'✓' if ok else '– outside bbox'} {name}")
                break
            except (OSError, IOError, RuntimeError, KeyError, ValueError) as err:
                if attempt < RETRIES:
                    log.warning(
                        "%s attempt %d failed (%s); retrying in %ds", name, attempt, err, RETRY_WAIT_S
                    )
                    time.sleep(RETRY_WAIT_S)
                else:
                    print(f"[{i}/{len(results)}] ✗ {name}: {err}")
                    failures.append({"granule": name, "error": str(err)})
        if i % 10 == 0:
            _save(zs, out / f"zone_spectra{suffix}.csv")
            _save(ss, out / f"station_values{suffix}.csv")
    if failures:
        pd.DataFrame(failures).to_csv(out / f"failed_granules{suffix}.csv", index=False)
        log.warning("%d granules failed; see failed_granules%s.csv", len(failures), suffix)
    return _save(zs, out / f"zone_spectra{suffix}.csv"), _save(ss, out / f"station_values{suffix}.csv")
