"""RQ3 groundwork: does surface chlorophyll lead bottom dissolved oxygen, and a blue-tide index.

Works on the in-situ long table alone (monitoring posts measure both surface Chl-a and bottom DO
hourly), so it can run before any assimilation exists. Satellite Chl-a can be substituted for the
surface series once matchups are dense enough.

  daily_series       hourly long table → daily means per station: chl_surface, do_surface, do_bottom, temp_surface,
                     temp_bottom, stratification (bottom = deepest layer within BOTTOM_TOL_M of the station depth)
  monthly_bottom     per station and month: hours observed, median bottom DO, share of hours below 3 / 2 / 1 mg/L
  station_summary    per station: days below 2 mg/L, events, minimum, chl→DO best lag, stratification correlation
  figure_bottom      four-row figure: DO (surface, bottom), stratification, surface chlorophyll
  lagged_correlation cross-correlation of chl(t) with do(t+lag), lag 0..max_lag days, per station and summer season
  hypoxia_events     runs of days with bottom DO below a threshold (the environmental standard uses 底層DO classes;
                     3 mg/L is a common hypoxia line for Tokyo Bay fisheries)
  event_lead_time    for each event, the last surface chl peak before onset → lead time in days
  blue_tide_index    sulfur proxy from Rrs(660), Higa et al. 2020 / 2021 symposium relation
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HYPOXIA_MGL = 3.0  # bottom-DO environmental standard, class 生物2
SURFACE_MAX_M = 3.0  # fixed ~2 m sensors report up to ~2.6 m at high tide
BOTTOM_TOL_M = 1.5


def _layers(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Boolean masks for the surface and bottom layer of a long table (labels first, depth as fallback).

    Bottom = label 'lower'/'bottom', or the deepest layer within BOTTOM_TOL_M of the station's depth
    (99th percentile of reported depths, which ignores occasional spurious values). Monitoring posts
    with fixed sensors (e.g. 2 / 11 / 20 m) report depths that move with the tide, hence the tolerance.
    """
    depth = d["depth_m"] if "depth_m" in d else pd.Series(np.nan, index=d.index)
    top = d["layer"].isin(["surface", "upper"]) | (d["layer"].isna() & (depth.fillna(0) <= SURFACE_MAX_M))
    bot = d["layer"].isin(["lower", "bottom"])
    if depth.notna().any():
        zmax = depth.groupby(d["station"]).transform(lambda z: z.quantile(0.99))
        bot = bot | (d["layer"].isna() & (depth >= zmax - BOTTOM_TOL_M) & (depth > SURFACE_MAX_M))
    return top, bot


def daily_series(insitu: pd.DataFrame) -> pd.DataFrame:
    """Daily means per station of the series needed here (QC 'ok' only; see `_layers` for the layer rule)."""
    d = insitu[(insitu["qc"] == "ok") & insitu["value"].notna()].copy()
    d["time"] = pd.to_datetime(d["time"], utc=True).dt.tz_convert("Asia/Tokyo")
    d["date"] = d["time"].dt.floor("D")
    top, bot = _layers(d)
    parts = {
        "chl_surface": (d[top & (d["param"] == "chl")], "first"),
        "do_surface": (d[top & (d["param"] == "do")], "first"),
        "temp_surface": (d[top & (d["param"] == "water_temp")], "first"),
        "do_bottom": (d[bot & (d["param"] == "do")], "last"),
        "temp_bottom": (d[bot & (d["param"] == "water_temp")], "last"),
    }
    out = None
    for name, (part, pick) in parts.items():
        # one value per timestamp: the shallowest layer for the surface, the deepest for the bottom
        one = part.sort_values("depth_m", na_position="last").groupby(["station", "time"], as_index=False)
        one = one.first() if pick == "first" else one.last()
        g = one.groupby(["station", "date"])["value"].mean().rename(name)
        out = g.to_frame() if out is None else out.join(g, how="outer")
    out = out.reset_index().sort_values(["station", "date"])
    out["stratification"] = out["temp_surface"] - out["temp_bottom"]
    return out


def monthly_bottom(insitu: pd.DataFrame) -> pd.DataFrame:
    """Per station and month: hours with a bottom-DO value, its median, share of hours below 3 / 2 / 1 mg/L, minimum."""
    d = insitu[(insitu["qc"] == "ok") & (insitu["param"] == "do") & insitu["value"].notna()].copy()
    d["time"] = pd.to_datetime(d["time"], utc=True).dt.tz_convert("Asia/Tokyo")
    _, bot = _layers(d)
    b = d[bot].sort_values("depth_m").groupby(["station", "time"], as_index=False).last()
    b["month"] = b["time"].dt.month
    g = b.groupby(["station", "month"])["value"]
    return pd.DataFrame(
        {
            "hours": g.size(),
            "median_mgl": g.median().round(2),
            "pct_below_3": (100 * g.apply(lambda v: (v < 3).mean())).round(0),
            "pct_below_2": (100 * g.apply(lambda v: (v < 2).mean())).round(0),
            "pct_below_1": (100 * g.apply(lambda v: (v < 1).mean())).round(0),
            "min_mgl": g.min().round(2),
        }
    ).reset_index()


def station_summary(daily: pd.DataFrame, threshold: float = 2.0, max_lag: int = 30) -> pd.DataFrame:
    """One row per station: coverage, days and events below `threshold`, minimum, chl→DO best lag, r with stratification."""
    ev = hypoxia_events(daily, threshold)
    lc = lagged_correlation(daily, max_lag=max_lag, months=tuple(range(1, 13)))
    rows = []
    for stn, g in daily.groupby("station"):
        y = g.set_index("date")["do_bottom"]
        e = ev[ev["station"] == stn] if not ev.empty else ev
        l0 = lc[(lc["station"] == stn) & (lc["lag_days"] == 0)] if not lc.empty else lc
        best = lc[lc["station"] == stn].iloc[0] if not lc.empty and (lc["station"] == stn).any() else None
        ok = g["stratification"].notna() & g["do_bottom"].notna()
        rows.append(
            {
                "station": stn,
                "days_observed": int(y.notna().sum()),
                f"days_below_{threshold:g}": int((y < threshold).sum()),
                "events_ge3d": len(e),
                "longest_event_d": int(e["days"].max()) if len(e) else 0,
                "min_daily_do": round(float(y.min()), 2) if y.notna().any() else np.nan,
                "date_min": y.idxmin().date() if y.notna().any() else pd.NaT,
                "r_chl_do_lag0": round(float(l0["r"].iloc[0]), 2) if len(l0) else np.nan,
                "best_lag_d": int(best["best_lag"]) if best is not None else np.nan,
                "r_best": round(float(best["best_r"]), 2) if best is not None else np.nan,
                "r_stratification_do": (
                    round(float(np.corrcoef(g.loc[ok, "stratification"], g.loc[ok, "do_bottom"])[0, 1]), 2)
                    if ok.sum() > 10
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def figure_bottom(daily: pd.DataFrame, summary: pd.DataFrame, out, marks: dict | None = None):
    """Rows = stations; columns = DO (surface and bottom), stratification, surface chlorophyll. Returns the path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e9e9e9",
        }
    )
    stations = list(summary["station"])
    fig, ax = plt.subplots(
        len(stations),
        3,
        figsize=(13, 2.3 * len(stations)),
        sharex=True,
        squeeze=False,
        gridspec_kw={"width_ratios": [2.2, 1, 1]},
    )
    for i, st in enumerate(stations):
        g = daily[daily["station"] == st].set_index("date")
        s = summary[summary["station"] == st].iloc[0]
        a = ax[i, 0]
        a.axhspan(0, 2, color="#c0392b", alpha=0.07, lw=0)
        a.axhspan(2, 3, color="#e07a2f", alpha=0.07, lw=0)
        a.plot(g.index, g["do_surface"], color="#9ec5e8", lw=1, label="surface DO")
        a.plot(g.index, g["do_bottom"], color="#1f3f6e", lw=1.4, label="bottom DO")
        a.axhline(3, color="#e07a2f", lw=0.7, ls="--")
        a.axhline(2, color="#c0392b", lw=0.7, ls="--")
        a.set_ylim(0, 14)
        a.set_ylabel("DO (mg/L), daily mean")
        a.set_title(st, loc="left", fontweight="bold")
        if i == 0:
            a.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2)
        b = ax[i, 1]
        b.plot(g.index, g["stratification"], color="#7b4fa6", lw=1.2)
        b.set_ylabel("ΔT surface − bottom (°C)")
        b.set_title(
            f"Stratification   r(ΔT, bottom DO) = {s['r_stratification_do']:+.2f}", loc="left", fontsize=8.5
        )
        c = ax[i, 2]
        c.plot(g.index, g["chl_surface"], color="#2a9d8f", lw=1.1)
        c.set_ylabel("Chl-a, surface (µg/L)")
        c.set_title(
            f"Surface chlorophyll   r(Chl → DO) = {s['r_best']:+.2f} at lag {s['best_lag_d']} d",
            loc="left",
            fontsize=8.5,
        )
        for axx in (a, b, c):
            for label, when in (marks or {}).items():
                axx.axvline(pd.Timestamp(when), color="#555", lw=0.7, ls=":")
    for axx in ax[-1]:
        axx.xaxis.set_major_locator(mdates.MonthLocator())
        axx.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout()
    from pathlib import Path

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def lagged_correlation(
    daily: pd.DataFrame, max_lag: int = 30, months=(5, 6, 7, 8, 9, 10), min_n: int = 40
) -> pd.DataFrame:
    """Pearson r between surface chl at day t and bottom DO at day t+lag. Negative r at positive lag = chl leads a DO drop."""
    rows = []
    for stn, g in daily.groupby("station"):
        g = g.set_index("date").asfreq("D")
        g = g[g.index.month.isin(months)]
        for lag in range(0, max_lag + 1):
            x, y = g["chl_surface"], g["do_bottom"].shift(-lag)
            ok = x.notna() & y.notna()
            if ok.sum() < min_n:
                continue
            rows.append(
                {
                    "station": stn,
                    "lag_days": lag,
                    "n": int(ok.sum()),
                    "r": float(np.corrcoef(x[ok], y[ok])[0, 1]),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    best = out.loc[out.groupby("station")["r"].idxmin(), ["station", "lag_days", "r"]].rename(
        columns={"lag_days": "best_lag", "r": "best_r"}
    )
    return out.merge(best, on="station")


def hypoxia_events(daily: pd.DataFrame, threshold: float = HYPOXIA_MGL, min_days: int = 3) -> pd.DataFrame:
    """Find runs of at least `min_days` consecutive days with bottom DO below `threshold`, per station."""
    rows = []
    for stn, g in daily.groupby("station"):
        g = g.set_index("date").asfreq("D")
        low = (g["do_bottom"] < threshold).fillna(False).to_numpy()
        i = 0
        while i < len(low):
            if low[i]:
                j = i
                while j + 1 < len(low) and low[j + 1]:
                    j += 1
                if j - i + 1 >= min_days:
                    rows.append(
                        {
                            "station": stn,
                            "onset": g.index[i],
                            "end": g.index[j],
                            "days": j - i + 1,
                            "min_do": float(np.nanmin(g["do_bottom"].iloc[i : j + 1])),
                        }
                    )
                i = j + 1
            else:
                i += 1
    return pd.DataFrame(rows)


def event_lead_time(
    daily: pd.DataFrame, events: pd.DataFrame, window: int = 30, peak_quantile: float = 0.9
) -> pd.DataFrame:
    """Days between the last surface-chl peak (above the station's seasonal 90th percentile) and hypoxia onset."""
    rows = []
    for ev in events.itertuples(index=False):
        g = daily[daily["station"] == ev.station].set_index("date")["chl_surface"].asfreq("D")
        thr = g.quantile(peak_quantile)
        before = g[(g.index < ev.onset) & (g.index >= ev.onset - pd.Timedelta(days=window))]
        peaks = before[before >= thr]
        rows.append(
            {
                "station": ev.station,
                "onset": ev.onset,
                "days": ev.days,
                "chl_peak_date": peaks.index.max() if len(peaks) else pd.NaT,
                "lead_days": (ev.onset - peaks.index.max()).days if len(peaks) else np.nan,
                "chl_peak": float(peaks.max()) if len(peaks) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def blue_tide_index(rrs660: np.ndarray, a: float = 0.753, b: float = 2.02) -> np.ndarray:
    """Elemental-sulfur proxy S0 (mg S/L) = a*log10(Rrs(660)) + b.

    Coefficients from the GOCI-based relation shown by Higa (2021 OPRI symposium; Higa et al. 2020 ECSS).
    They were fitted to GOCI Rrs in Tokyo Bay and must be recalibrated for PACE before any quantitative use;
    until then treat the index as relative (high = suspected blue tide, together with high turbidity and low Chl-a).
    """
    rrs660 = np.asarray(rrs660, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(rrs660 > 0, a * np.log10(rrs660) + b, np.nan)
