"""Flags, store, baselines, OWT, models, transfer, lag analysis, export — on small synthetic inputs."""

import numpy as np
import pandas as pd
import pytest

from dualbay import baseline, db, export, flags, lag, models, owt, transfer
from dualbay.demo import WL, toy_rrs


def test_flags_mask_and_fractions():
    f = np.zeros((4, 4), dtype=np.int32)
    f[0, 0] |= 1 << flags.L2_FLAG_BITS["LAND"]
    f[1, 1] |= 1 << flags.L2_FLAG_BITS["ABSAER"]
    f[2, 2] |= 1 << flags.L2_FLAG_BITS["TURBIDW"]  # not in the standard mask
    good = flags.good_pixels(f, f.shape, "standard")
    # ABSAER is deliberately NOT masked (set on nearly every pixel over the bays); LAND is
    assert good.sum() == 15 and not good[0, 0] and good[1, 1] and good[2, 2]
    assert flags.good_pixels(f, f.shape, "standard", extra=("ABSAER",)).sum() == 14
    fr = flags.flag_fractions(f, np.ones((4, 4), bool))
    assert fr["frac_absaer"] == pytest.approx(1 / 16) and fr["frac_turbidw"] == pytest.approx(1 / 16)
    # bit table read from the file's own flag_meanings attribute (PACE OCI L2 string)
    fm = (
        "ATMFAIL LAND PRODWARN HIGLINT HILT HISATZEN COASTZ SPARE STRAYLIGHT CLDICE COCCOLITH TURBIDW HISOLZEN SPARE "
        "LOWLW CHLFAIL NAVWARN ABSAER SPARE MAXAERITER MODGLINT CHLWARN ATMWARN OPSHAL SEAICE NAVFAIL FILTER SPARE "
        "BOWTIEDEL HIPOL PRODFAIL SPARE"
    )
    masks = [1 << i for i in range(31)] + [-2147483648]
    tb = flags.from_file_attrs(fm, masks)
    assert tb["ABSAER"] == 17 and tb["OPSHAL"] == 23 and tb["NAVFAIL"] == 25 and "SPARE" not in tb
    assert tb == {k: v for k, v in flags.L2_FLAG_BITS.items()}


def _spectra(n, chl, rng, bay="tokyo", year=2025):
    rrs = toy_rrs(chl, np.full(n, 6.0), np.full(n, 0.4), rng)
    df = pd.DataFrame(rrs, columns=[f"rrs_{w}" for w in WL])
    df["y"] = chl
    df["bay"] = bay
    df["scene_time"] = pd.to_datetime(pd.date_range(f"{year}-01-01", periods=n, freq="D", tz="UTC"))
    return df


def test_baseline_tiers_and_metrics():
    rng = np.random.default_rng(1)
    train = _spectra(120, rng.lognormal(np.log(12), 0.6, 120), rng)
    test = _spectra(60, rng.lognormal(np.log(12), 0.6, 60), rng)
    tiers = baseline.run_tiers(train, test, "chl")
    ndci = tiers[tiers["algorithm"] == "ndci"].set_index("tier")
    assert ndci.loc[2, "rmse"] < ndci.loc[1, "rmse"]  # local recalibration helps
    m = baseline.metrics([1, 2, 4], [1.1, 1.8, 4.4], [0.9, 1.5, 3], [1.3, 2.5, 5])
    assert m["n"] == 3 and m["picp68"] == 1.0 and 0 < m["mdsa"] < 15


def test_owt_and_model_uncertainty():
    rng = np.random.default_rng(2)
    a = _spectra(150, rng.lognormal(np.log(15), 0.5, 150), rng)
    b = _spectra(150, rng.lognormal(np.log(0.8), 0.4, 150), rng, bay="sagami")
    both = pd.concat([a, b], ignore_index=True)
    scheme = owt.OWTScheme(k_min=2, k_max=4).fit(both).fit_thresholds(both)
    lab = scheme.predict(both)
    assert scheme.k >= 2 and lab["owt"].nunique() == scheme.k
    # the two bays should mostly fall into different types
    assert (lab.loc[:149, "owt"].mode()[0]) != (lab.loc[150:, "owt"].mode()[0])
    both["owt"] = lab["owt"]
    rf = models.RFRetrieval(n_estimators=100).fit(both.iloc[::2], both["y"].iloc[::2])
    pred = rf.predict(both.iloc[1::2])
    assert (pred["y_lo"] <= pred["y_pred"]).all() and (pred["y_pred"] <= pred["y_hi"]).all()
    met = baseline.metrics(both["y"].iloc[1::2], pred["y_pred"], pred["y_lo"], pred["y_hi"])
    assert met["r2"] > 0.8 and 0.4 < met["picp68"] < 0.95


def test_transfer_experiments():
    rng = np.random.default_rng(3)
    frames = [_spectra(150, rng.lognormal(np.log(15), 0.5, 150), rng, year=y) for y in (2024, 2025, 2026)]
    tk = pd.concat(frames, ignore_index=True)
    sg = _spectra(80, rng.lognormal(np.log(0.8), 0.4, 80), rng, bay="sagami", year=2025)
    th = transfer.temporal_holdout(tk, [2024, 2025], 2026)
    assert "all" in set(th["subset"]) and th.loc[th["subset"] == "all", "n"].iloc[0] == 150
    sva = transfer.static_vs_adaptive(tk, [2024, 2025], 2026)
    assert (
        len(sva) == tk[tk["scene_time"].dt.year == 2026]["scene_time"].dt.month.nunique()
        and sva["n_new_used"].iloc[0] == 0
    )
    sp = transfer.spatial_transfer(pd.concat([tk, sg], ignore_index=True), "tokyo", "sagami")
    assert sp["share_novel"] > 0.5  # clean water is a new optical type for a Tokyo model
    lc = transfer.learning_curve(
        pd.concat([tk, sg], ignore_index=True), "tokyo", "sagami", sizes=(0, 10, 30), repeats=2
    )
    r = lc[("rmse", "mean")]
    assert r.iloc[-1] < r.iloc[0]  # local samples help


def test_lag_analysis_recovers_planted_lag():
    rng = np.random.default_rng(4)
    days = pd.date_range("2025-05-01", "2025-10-31", freq="D", tz="Asia/Tokyo")
    chl = 10 * rng.lognormal(0, 0.3, len(days))
    chl[60:66] *= 5
    chl[120:126] *= 4
    do = 6 - 0.08 * np.roll(np.convolve(chl, np.ones(3) / 3, "same"), 7) + rng.normal(0, 0.2, len(days))
    rows = []
    for i, d in enumerate(days):
        rows.append(("s1", d, "surface", 0.5, "chl", chl[i], "ug/L", "ok", "t"))
        rows.append(("s1", d, "bottom", 10.0, "do", do[i], "mg/L", "ok", "t"))
    ins = pd.DataFrame(
        rows, columns=["station", "time", "layer", "depth_m", "param", "value", "unit", "qc", "source_file"]
    )
    daily = lag.daily_series(ins)
    lc = lag.lagged_correlation(daily, max_lag=20, min_n=30)
    assert 5 <= lc["best_lag"].iloc[0] <= 9
    ev = lag.hypoxia_events(daily, threshold=3.0, min_days=2)
    assert len(ev) >= 1
    lt = lag.event_lead_time(daily, ev)
    assert lt["lead_days"].notna().any()
    bt = lag.blue_tide_index(np.array([0.001, 0.01, -1]))
    assert np.isnan(bt[2]) and bt[1] > bt[0]


def test_store_and_export(tmp_path):
    dbp = tmp_path / "t.sqlite"
    ins = pd.DataFrame(
        {
            "station": ["kawasaki_island"] * 2,
            "time": pd.to_datetime(["2025-07-14 09:00", "2025-07-14 10:00"]).tz_localize("Asia/Tokyo"),
            "layer": ["upper", "upper"],
            "depth_m": [np.nan, np.nan],
            "param": ["chl", "chl"],
            "value": [21.5, 19.8],
            "unit": ["ug/L"] * 2,
            "qc": ["ok"] * 2,
            "source_file": ["f"] * 2,
        }
    )
    with db.connect(dbp) as con:
        db.load_dimensions(con)
        assert db.load_insitu(con, ins) == 2
        assert db.load_insitu(con, ins) == 0  # idempotent
        with pytest.raises(ValueError):
            db.load_insitu(con, ins.assign(station="nowhere"))
        run = db.new_run(con, "ocx", "chl", 1)
        db.save_metrics(con, run, "all", {"n": 2, "rmse": 1.0})
    files = export.export(dbp, tmp_path)
    names = {p.name for p in files}
    assert {"DimDate.csv", "FactInsituDaily.csv", "FactMetric.csv", "measures.dax"} <= names
    fd = pd.read_csv(tmp_path / "powerbi" / "FactInsituDaily.csv")
    assert fd["MeanValue"].iloc[0] == pytest.approx(20.65)


def test_bottom_layer_rule_and_monthly_table():
    """Fixed sensors at 2 / 11 / 20 m whose reported depth moves with the tide: bottom = deepest within 1.5 m."""
    t = pd.date_range("2025-07-01", periods=48, freq="h", tz="Asia/Tokyo")
    rows = []
    for i, ti in enumerate(t):
        tide = 0.6 * np.sin(i / 12 * np.pi)
        for z, do_v, temp in ((2.0, 8.0, 27.0), (11.0, 5.0, 23.0), (20.0, 1.0 + (i % 5) * 0.1, 19.0)):
            rows.append(("s1", ti, np.nan, z + tide, "do", do_v, "mg/L", "ok"))
            rows.append(("s1", ti, np.nan, z + tide, "water_temp", temp, "degC", "ok"))
            rows.append(("s1", ti, np.nan, z + tide, "chl", 20.0, "ug/L", "ok"))
    ins = pd.DataFrame(rows, columns=["station", "time", "layer", "depth_m", "param", "value", "unit", "qc"])
    daily = lag.daily_series(ins)
    assert len(daily) == 2
    assert daily["do_bottom"].between(1.0, 1.5).all() and daily["do_surface"].eq(8.0).all()
    assert daily["stratification"].eq(8.0).all()
    mb = lag.monthly_bottom(ins)
    assert mb.loc[0, "pct_below_2"] == 100 and mb.loc[0, "pct_below_3"] == 100 and mb.loc[0, "hours"] == 48
    summ = lag.station_summary(daily)
    assert summ.loc[0, "days_below_2"] == 2


def test_rededge_indices_and_cross_validation():
    from dualbay import rededge

    rng = np.random.default_rng(1)
    n = 60
    chl = 10 ** rng.uniform(0.5, 1.8, n)
    ndci = 0.4 * (np.log10(chl) - 1.2) + rng.normal(0, 0.05, n)  # a red edge that tracks chlorophyll
    r665 = 0.002 * np.ones(n)
    r709 = r665 * (1 + ndci) / (1 - ndci)
    t = pd.date_range("2025-07-01 03:00", periods=n, freq="D", tz="UTC")
    stations = np.array(["a", "b", "c"])[np.arange(n) % 3]
    long = []
    for i in range(n):
        for nm, v in (
            (443, 0.004),
            (490, 0.0045),
            (510, 0.004),
            (560, 0.005),
            (620, 0.003),
            (665, r665[i]),
            (681, 0.0022),
            (709, r709[i]),
            (719, 0.0015),
        ):
            long.append((t[i], f"g{i}", stations[i], 0.5, 9, True, float(nm), v, 0.5 * v))
    sv = pd.DataFrame(
        long,
        columns=[
            "time",
            "granule",
            "station",
            "dist_km",
            "n_valid",
            "passed",
            "wavelength",
            "rrs",
            "rrs_unc",
        ],
    )
    sp = rededge.indices(rededge.window_spectra(sv))
    assert len(sp) == n and np.allclose(sp["ndci"], ndci, atol=1e-9) and sp["ndci_unc"].gt(0).all()
    ins = pd.DataFrame(
        {
            "station": np.repeat(stations, 3),
            "time": np.repeat(t, 3) + pd.to_timedelta(np.tile([-1, 0, 1], n), unit="h"),
            "layer": np.nan,
            "depth_m": 1.0,
            "param": "chl",
            "value": np.repeat(chl, 3),
            "unit": "ug/L",
            "qc": "ok",
        }
    )
    m = rededge.matchups(sp, ins, window_h=1.5)
    assert len(m) == n
    table, m2 = rededge.evaluate(m)
    loso = table[table["estimate"].str.startswith("ndci, local linear fit, leave-one-station")].iloc[0]
    control = table[table["estimate"].str.startswith("control")].iloc[0]
    assert loso["mdsa_pct"] < control["mdsa_pct"] and loso["r2_log"] > 0.8
    assert table.attrs["correlation_with_log_chl"]["ndci"] > 0.9
    rel = rededge.relative_uncertainty(sv)
    assert np.allclose(rel["median"], 0.5)
