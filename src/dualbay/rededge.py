"""Red-edge test on PACE OCI station windows: can the 665/709 nm bands, which the standard
chlorophyll algorithm does not use, recover chlorophyll where the blue–green ratio fails?

Input: `station_values.csv` from `dualbay pace` (AOP product: window median Rrs and Rrs_unc per
wavelength) and the in-situ long table from `dualbay parse-insitu`. Optionally `station_values_bgc.csv`
for the standard chlor_a at the same windows.

  window_spectra      long station table → one row per (time, station) with the bands used here
  indices             NDCI, two-band ratio, fluorescence line height, blue–green ratio, propagated NDCI uncertainty
  matchups            pair window spectra with the station's surface chlorophyll within ±window_h
  evaluate            published NDCI, local linear fits (leave-one-station-out, leave-one-month-out),
                      a random forest on all bands, the constant-median control, and chlor_a on the common subset
  relative_uncertainty median Rrs_unc/Rrs per wavelength over the station windows
  figure              three panels: index vs chlorophyll, predicted vs observed, uncertainty-to-signal spectrum

Everything is in log10 space, as in the ocean-colour literature; metrics come from `baseline.metrics`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import baseline

BANDS = (443, 490, 510, 560, 620, 665, 681, 709, 719)  # nm; all inside the standard 346–719 nm range
BAND_TOL = 1.6  # nm; PACE OCI bands are ~2.5 nm apart in the visible
NDCI_PUBLISHED = (14.039, 86.115, 194.325)  # Mishra & Mishra (2012): Chl = a + b·NDCI + c·NDCI²
SURFACE_MAX_DEPTH_M = 3.0  # shallowest layer; fixed ~2 m sensors report up to ~2.6 m at high tide


def window_spectra(station_values: pd.DataFrame, min_valid: int = 5) -> pd.DataFrame:
    """One row per (time, station): the band values in BANDS (mean over ±BAND_TOL nm) and their Rrs_unc."""
    sv = station_values[station_values["passed"] & (station_values["n_valid"] >= min_valid)]
    cols = {}
    for nm in BANDS:
        sel = sv[(sv["wavelength"] - nm).abs() <= BAND_TOL]
        g = sel.groupby(["time", "station"])
        cols[nm] = g["rrs"].mean()
        if "rrs_unc" in sv:
            cols[f"unc_{nm}"] = g["rrs_unc"].mean()
    out = pd.DataFrame(cols).dropna(subset=list(BANDS)).reset_index()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    return out


def indices(sp: pd.DataFrame) -> pd.DataFrame:
    """Add the four candidate features and the NDCI uncertainty propagated from Rrs_unc (independent errors)."""
    sp = sp.copy()
    r665, r709, r681 = sp[665], sp[709], sp[681]
    num, den = r709 - r665, r709 + r665
    sp["ndci"] = num / den
    sp["ratio_709_665"] = r709 / r665
    # fluorescence line height: 681 nm against the straight line between 665 and 709 nm
    sp["flh"] = r681 - (r665 + (r709 - r665) * (681 - 665) / (709 - 665))
    with np.errstate(divide="ignore", invalid="ignore"):
        sp["blue_green"] = np.log10(sp[[443, 490, 510]].max(axis=1) / sp[560])
    if "unc_665" in sp and "unc_709" in sp:
        sp["ndci_unc"] = np.sqrt(
            (2 * r665 / den**2 * sp["unc_709"]) ** 2 + (2 * r709 / den**2 * sp["unc_665"]) ** 2
        )
    return sp


def surface_chl(insitu: pd.DataFrame) -> pd.DataFrame:
    """Shallowest valid chlorophyll value per (station, time): the layer a satellite sees."""
    d = insitu[(insitu["param"] == "chl") & (insitu["qc"] == "ok") & insitu["value"].notna()].copy()
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d = d[d["depth_m"].isna() | (d["depth_m"] <= SURFACE_MAX_DEPTH_M)]
    d = d.sort_values("depth_m", na_position="last").groupby(["station", "time"], as_index=False).first()
    return d[["station", "time", "value"]].rename(columns={"value": "insitu"})


def matchups(sp: pd.DataFrame, insitu: pd.DataFrame, window_h: float = 1.5, min_obs: int = 2) -> pd.DataFrame:
    """Pair each window spectrum with the median station chlorophyll within ±window_h hours."""
    chl = surface_chl(insitu)
    rows = []
    for _, r in sp.iterrows():
        s = chl[
            (chl["station"] == r["station"])
            & ((chl["time"] - r["time"]).abs() <= pd.Timedelta(hours=window_h))
        ]
        if len(s) < min_obs:
            continue
        rows.append({**r.to_dict(), "insitu": float(s["insitu"].median()), "n_insitu": len(s)})
    m = pd.DataFrame(rows)
    if m.empty:
        return m
    m = m[(m[665] > 0) & (m[709] > 0) & (m["insitu"] > 0)].copy()
    m["log_chl"] = np.log10(m["insitu"])
    m["month"] = m["time"].dt.month
    return m.reset_index(drop=True)


def _cv_linear(m: pd.DataFrame, feature: str, group: str) -> pd.Series:
    """Leave-one-group-out prediction of chlorophyll from a linear fit of log10(Chl) on the feature."""
    pred = pd.Series(np.nan, index=m.index)
    for g in m[group].unique():
        tr, te = m[m[group] != g], m[m[group] == g]
        if len(tr) < 5:
            continue
        b, a = np.polyfit(tr[feature], tr["log_chl"], 1)
        pred[te.index] = 10 ** (a + b * te[feature])
    return pred


def _cv_forest(m: pd.DataFrame, group: str, seed: int = 0) -> pd.Series:
    """Leave-one-group-out random forest on log10 of all bands."""
    from sklearn.ensemble import RandomForestRegressor

    X = np.log10(m[list(BANDS)].clip(lower=1e-5))
    pred = pd.Series(np.nan, index=m.index)
    for g in m[group].unique():
        tr = m[group] != g
        if tr.sum() < 10:
            continue
        rf = RandomForestRegressor(300, min_samples_leaf=5, random_state=seed).fit(
            X[tr], m.loc[tr, "log_chl"]
        )
        pred[~tr] = 10 ** rf.predict(X[~tr])
    return pred


def _row(name: str, y, yhat) -> dict:
    k = baseline.metrics(y, yhat)
    return {
        "estimate": name,
        "n": k.get("n", 0),
        "bias_pct": round(100 * (k["log_bias"] - 1)) if "log_bias" in k else np.nan,
        "mdsa_pct": round(k["mdsa"]) if "mdsa" in k else np.nan,
        "r2_log": _r2_log(y, yhat),
    }


def _r2_log(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat) & (y > 0) & (yhat > 0)
    if ok.sum() < 3:
        return np.nan
    ly, lh = np.log10(y[ok]), np.log10(yhat[ok])
    return round(float(1 - np.sum((lh - ly) ** 2) / np.sum((ly - ly.mean()) ** 2)), 2)


def evaluate(m: pd.DataFrame, chlor_a: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Accuracy table for every estimate, and the matchup table with the cross-validated NDCI prediction added."""
    m = m.copy()
    a, b, c = NDCI_PUBLISHED
    rows = [
        _row(
            "NDCI, published coefficients (Mishra & Mishra 2012)",
            m["insitu"],
            a + b * m["ndci"] + c * m["ndci"] ** 2,
        )
    ]
    for f in ("ndci", "ratio_709_665", "flh", "blue_green"):
        m[f"{f}_loso"] = _cv_linear(m, f, "station")
        rows.append(_row(f"{f}, local linear fit, leave-one-station-out", m["insitu"], m[f"{f}_loso"]))
    m["ndci_lomo"] = _cv_linear(m, "ndci", "month")
    rows.append(_row("ndci, local linear fit, leave-one-month-out", m["insitu"], m["ndci_lomo"]))
    m["rf_loso"] = _cv_forest(m, "station")
    rows.append(_row(f"random forest, {len(BANDS)} bands, leave-one-station-out", m["insitu"], m["rf_loso"]))
    const = pd.Series(np.nan, index=m.index)
    for st in m["station"].unique():
        const[m["station"] == st] = 10 ** m.loc[m["station"] != st, "log_chl"].median()
    rows.append(_row("control: seasonal median of the other stations", m["insitu"], const))
    if chlor_a is not None and not chlor_a.empty:
        mb = m.merge(chlor_a[["time", "station", "chlor_a"]], on=["time", "station"])
        rows.append(_row("NASA standard chlor_a (BGC), same windows", mb["insitu"], mb["chlor_a"]))
        rows.append(_row("ndci leave-one-station-out, chlor_a subset", mb["insitu"], mb["ndci_loso"]))
    corr = {
        f: round(float(np.corrcoef(m[f], m["log_chl"])[0, 1]), 2)
        for f in ("ndci", "ratio_709_665", "flh", "blue_green")
    }
    table = pd.DataFrame(rows)
    table.attrs["correlation_with_log_chl"] = corr
    return table, m


def relative_uncertainty(station_values: pd.DataFrame, min_valid: int = 5) -> pd.DataFrame:
    """Median and quartiles of Rrs_unc/Rrs per wavelength over all passing station windows (positive Rrs only)."""
    sv = station_values[
        station_values["passed"] & (station_values["n_valid"] >= min_valid) & (station_values["rrs"] > 0)
    ]
    rel = (sv["rrs_unc"] / sv["rrs"]).rename("rel")
    g = pd.concat([sv["wavelength"], rel], axis=1).groupby("wavelength")["rel"]
    return pd.DataFrame(
        {"median": g.median(), "p25": g.quantile(0.25), "p75": g.quantile(0.75)}
    ).reset_index()


def figure(
    m: pd.DataFrame, rel: pd.DataFrame, table: pd.DataFrame, out: Path, chlor_a: pd.DataFrame | None = None
):
    """Three-panel figure; returns the written path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e6e6e6",
        }
    )
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.1))
    for st, g in m.groupby("station"):
        ax[0].scatter(g["ndci"], g["insitu"], s=22, alpha=0.8, label=st, edgecolor="white", linewidth=0.5)
    b, a = np.polyfit(m["ndci"], m["log_chl"], 1)
    x = np.linspace(m["ndci"].min(), m["ndci"].max(), 50)
    r = table.attrs.get("correlation_with_log_chl", {}).get("ndci", np.nan)
    ax[0].plot(x, 10 ** (a + b * x), color="#333", lw=1.2, label=f"fit, all pairs (r = {r:.2f})")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("NDCI = (Rrs709 − Rrs665)/(Rrs709 + Rrs665)")
    ax[0].set_ylabel("Station chlorophyll (µg/L)")
    ax[0].set_title(f"a  Red-edge index vs station chlorophyll, n = {len(m)}", loc="left")
    ax[0].legend(frameon=False, fontsize=7.5)

    lim = [0.8, 150]
    ax[1].plot(lim, lim, color="#999", lw=0.8, ls="--")
    if chlor_a is not None and not chlor_a.empty:
        mb = m.merge(chlor_a[["time", "station", "chlor_a"]], on=["time", "station"])
        ax[1].scatter(
            mb["insitu"],
            mb["chlor_a"],
            s=26,
            marker="s",
            color="#c0392b",
            alpha=0.75,
            label=f"NASA chlor_a (n = {len(mb)})",
            edgecolor="white",
            linewidth=0.5,
        )
    ax[1].scatter(
        m["insitu"],
        m["ndci_loso"],
        s=18,
        color="#2a9d8f",
        alpha=0.75,
        label=f"NDCI, local fit, leave-one-station-out (n = {len(m)})",
        edgecolor="white",
        linewidth=0.5,
    )
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlim(lim)
    ax[1].set_ylim(lim)
    ax[1].set_xlabel("Station chlorophyll (µg/L)")
    ax[1].set_ylabel("Satellite estimate (µg/L)")
    ax[1].set_title("b  Standard product vs recalibrated red edge", loc="left")
    ax[1].legend(frameon=False, fontsize=7, loc="upper left")

    ax[2].fill_between(rel["wavelength"], rel["p25"], rel["p75"], color="#1f5fa8", alpha=0.15, lw=0)
    ax[2].plot(rel["wavelength"], rel["median"], color="#1f5fa8", lw=1.4)
    ax[2].axhline(1, color="#c0392b", lw=0.9, ls="--")
    ax[2].text(350, 1.05, "uncertainty = signal", color="#c0392b", fontsize=7.5)
    for nm in (665, 709):
        ax[2].axvline(nm, color="#999", lw=0.6, ls=":")
    ax[2].set_yscale("log")
    ax[2].set_ylim(0.03, 3)
    ax[2].set_xlim(340, 725)
    ax[2].set_xlabel("Wavelength (nm)")
    ax[2].set_ylabel("Rrs_unc / Rrs (median, IQR shaded)")
    ax[2].set_title("c  NASA per-pixel uncertainty relative to signal", loc="left")
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def run(pace_dir: str | Path, insitu_csv: str | Path, out_dir: str | Path, window_h: float = 1.5) -> dict:
    """Whole chain from the CSV tables to `out_dir`: matchups, metrics, relative uncertainty and the figure."""
    pace_dir, out_dir = Path(pace_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sv = pd.read_csv(pace_dir / "station_values.csv", parse_dates=["time"])
    ins = pd.read_csv(insitu_csv, parse_dates=["time"])
    chlor_a = None
    bgc = pace_dir / "station_values_bgc.csv"
    if bgc.exists():
        b = pd.read_csv(bgc, parse_dates=["time"])
        b["time"] = pd.to_datetime(b["time"], utc=True)
        chlor_a = b[b["passed"]][["time", "station", "chlor_a"]].drop_duplicates(["time", "station"])
    m = matchups(indices(window_spectra(sv)), ins, window_h)
    if m.empty:
        raise SystemExit("No red-edge matchups: check station ids and the time window")
    table, m = evaluate(m, chlor_a)
    rel = relative_uncertainty(sv)
    table.to_csv(out_dir / "rededge_metrics.csv", index=False)
    m.to_csv(out_dir / "rededge_matchups.csv", index=False)
    rel.to_csv(out_dir / "relative_uncertainty_by_wavelength.csv", index=False)
    fig = figure(m, rel, table, out_dir / "rededge.png", chlor_a)
    return {
        "n": len(m),
        "metrics": table,
        "correlation": table.attrs["correlation_with_log_chl"],
        "figure": fig,
    }
