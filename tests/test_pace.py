"""PACE reader and figures on a small synthetic file with the real L2 group layout."""

import h5py
import numpy as np
import pandas as pd

from dualbay import figures, pace

WL = np.arange(400, 721, 5, dtype="float32")


def _spectrum(kind):
    """Rough shapes: green-peaked turbid eutrophic water vs blue clear water."""
    if kind == "tokyo":
        return 0.002 + 0.006 * np.exp(-(((WL - 565) / 45) ** 2)) + 0.0015 * np.exp(-(((WL - 700) / 12) ** 2))
    return 0.006 * np.exp(-(((WL - 440) / 60) ** 2)) + 0.0005


def _make_granule(path, cloudy_sagami=False):
    lat1d = np.linspace(34.8, 35.9, 120)
    lon1d = np.linspace(138.9, 140.3, 140)
    lon, lat = np.meshgrid(lon1d, lat1d)
    rrs = np.empty(lat.shape + (len(WL),), dtype="float32")
    tokyo = lon > 139.6
    rrs[tokyo] = _spectrum("tokyo")
    rrs[~tokyo] = _spectrum("sagami")
    flags = np.zeros(lat.shape, dtype="int32")
    flags[(lat > 35.75)] |= pace.LAND_BIT  # a strip of "land" in the north
    if cloudy_sagami:
        rrs[(lon < 139.55)] = np.nan
    scale, offset, fill = 2e-6, 0.05, -32767
    packed = np.round((rrs - offset) / scale)
    packed[np.isnan(packed)] = fill
    chl = np.where(tokyo, 15.0, 0.8).astype("float32")

    with h5py.File(path, "w") as f:
        f["navigation_data/latitude"] = lat.astype("float32")
        f["navigation_data/longitude"] = lon.astype("float32")
        f["sensor_band_parameters/wavelength_3d"] = WL
        d = f.create_dataset("geophysical_data/Rrs", data=packed.astype("int16"))
        d.attrs["scale_factor"] = np.float32(scale)
        d.attrs["add_offset"] = np.float32(offset)
        d.attrs["_FillValue"] = np.int16(fill)
        f["geophysical_data/l2_flags"] = flags
        c = f.create_dataset("geophysical_data/chlor_a", data=chl)
        c.attrs["_FillValue"] = np.float32(-32767.0)


def test_read_and_extract(tmp_path):
    p1 = tmp_path / "PACE_OCI.20250714T031500.L2.OC_AOP.V3_2.nc"
    p2 = tmp_path / "PACE_OCI.20250820T033000.L2.OC_AOP.V3_2.nc"
    _make_granule(p1)
    _make_granule(p2, cloudy_sagami=True)
    z, s = pace.process_files([p1, p2], tmp_path / "out")

    first = z[z["granule"] == p1.name].drop_duplicates("zone").set_index("zone")
    assert first.loc["tokyo_inner", "valid_frac"] == 1.0
    assert first.loc["sagami_open", "valid_frac"] == 1.0
    second = z[z["granule"] == p2.name].drop_duplicates("zone").set_index("zone")
    assert second.loc["sagami_open", "valid_frac"] == 0.0

    t = z[(z["granule"] == p1.name) & (z["zone"] == "tokyo_inner")]
    peak = t.loc[t["rrs_median"].idxmax(), "wavelength"]
    assert 550 <= peak <= 580  # scale/offset decoded correctly, green peak kept
    assert pace.granule_time(p1.name) == pd.Timestamp("2025-07-14 03:15:00", tz="UTC")

    kaw = s[(s["station"] == "kawasaki_island") & (s["granule"] == p1.name)]
    assert not kaw.empty and kaw["passed"].all() and (kaw["chlor_a"] == 15.0).all()


def test_figures(tmp_path):
    p = tmp_path / "PACE_OCI.20250714T031500.L2.OC_AOP.V3_2.nc"
    _make_granule(p)
    z, s = pace.process_files([p], tmp_path / "pace")
    s.to_csv(tmp_path / "pace" / "station_values_bgc.csv", index=False)
    made = figures.make_all(tmp_path / "pace", tmp_path / "fig")
    assert len(made) == 3 and all(m.exists() and m.stat().st_size > 10_000 for m in made)
