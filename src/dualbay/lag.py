"""RQ3 groundwork: does surface chlorophyll lead bottom dissolved oxygen, and a blue-tide index.

Works on the in-situ long table alone (monitoring posts measure both surface Chl-a and bottom DO
hourly), so it can run before any assimilation exists. Satellite Chl-a can be substituted for the
surface series once matchups are dense enough.

  daily_series       hourly long table → daily means per station: chl_surface, do_bottom, temp_surface, temp_bottom
  lagged_correlation cross-correlation of chl(t) with do(t+lag), lag 0..max_lag days, per station and summer season
  hypoxia_events     runs of days with bottom DO below a threshold (the environmental standard uses 底層DO classes;
                     3 mg/L is a common hypoxia line for Tokyo Bay fisheries)
  event_lead_time    for each event, the last surface chl peak before onset → lead time in days
  blue_tide_index    sulfur proxy from Rrs(660), Higa et al. 2020 / 2021 symposium relation
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HYPOXIA_MGL = 3.0


def daily_series(insitu: pd.DataFrame) -> pd.DataFrame:
    """Daily means per station for the series needed here. Bottom = layer 'lower'/'bottom' or depth >= 80 % of max."""
    d = insitu[insitu["qc"] == "ok"].copy()
    d["time"] = pd.to_datetime(d["time"], utc=True).dt.tz_convert("Asia/Tokyo")
    d["date"] = d["time"].dt.floor("D")
    top = d["layer"].isin(["surface", "upper"]) | (d["layer"].isna() & (d["depth_m"].fillna(0) <= 2))
    bot = d["layer"].isin(["lower", "bottom"])
    if "depth_m" in d and d["depth_m"].notna().any():
        dmax = d.groupby("station")["depth_m"].transform("max")
        bot = bot | (d["layer"].isna() & (d["depth_m"] >= 0.8 * dmax) & (d["depth_m"] > 2))
    parts = {
        "chl_surface": d[top & (d["param"] == "chl")],
        "temp_surface": d[top & (d["param"] == "water_temp")],
        "do_bottom": d[bot & (d["param"] == "do")],
        "temp_bottom": d[bot & (d["param"] == "water_temp")],
    }
    out = None
    for name, part in parts.items():
        g = part.groupby(["station", "date"])["value"].mean().rename(name)
        out = g.to_frame() if out is None else out.join(g, how="outer")
    out = out.reset_index().sort_values(["station", "date"])
    out["stratification"] = out["temp_surface"] - out["temp_bottom"]
    return out


def lagged_correlation(daily: pd.DataFrame, max_lag: int = 30, months=(5, 6, 7, 8, 9, 10), min_n: int = 40) -> pd.DataFrame:
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
            rows.append({"station": stn, "lag_days": lag, "n": int(ok.sum()), "r": float(np.corrcoef(x[ok], y[ok])[0, 1])})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    best = out.loc[out.groupby("station")["r"].idxmin(), ["station", "lag_days", "r"]].rename(columns={"lag_days": "best_lag", "r": "best_r"})
    return out.merge(best, on="station")


def hypoxia_events(daily: pd.DataFrame, threshold: float = HYPOXIA_MGL, min_days: int = 3) -> pd.DataFrame:
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
                    rows.append({"station": stn, "onset": g.index[i], "end": g.index[j], "days": j - i + 1,
                                 "min_do": float(np.nanmin(g["do_bottom"].iloc[i:j + 1]))})
                i = j + 1
            else:
                i += 1
    return pd.DataFrame(rows)


def event_lead_time(daily: pd.DataFrame, events: pd.DataFrame, window: int = 30, peak_quantile: float = 0.9) -> pd.DataFrame:
    """Days between the last surface-chl peak (above the station's seasonal 90th percentile) and hypoxia onset."""
    rows = []
    for ev in events.itertuples(index=False):
        g = daily[daily["station"] == ev.station].set_index("date")["chl_surface"].asfreq("D")
        thr = g.quantile(peak_quantile)
        before = g[(g.index < ev.onset) & (g.index >= ev.onset - pd.Timedelta(days=window))]
        peaks = before[before >= thr]
        rows.append({"station": ev.station, "onset": ev.onset, "days": ev.days,
                     "chl_peak_date": peaks.index.max() if len(peaks) else pd.NaT,
                     "lead_days": (ev.onset - peaks.index.max()).days if len(peaks) else np.nan,
                     "chl_peak": float(peaks.max()) if len(peaks) else np.nan})
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
