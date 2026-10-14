"""Summary figures from the PACE tables: scene availability, zone spectra, station chlorophyll.

fig_availability   clear PACE scenes per month, Tokyo Bay vs Sagami Bay
fig_spectra        median Rrs spectra per zone on clear days, absolute and shape-normalised
fig_station_chl    standard NASA chlor_a at monitoring posts over time
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ZONE_STYLE = {
    "tokyo_inner": ("Inner Tokyo Bay", "#b5651d"),
    "uraga_channel": ("Uraga Channel", "#d9a066"),
    "miura_transition": ("Miura transition", "#5b6f78"),
    "sagami_coastal": ("Sagami coastal", "#6fa882"),
    "sagami_open": ("Open Sagami Bay", "#2f7d4f"),
}
CLEAR_FRAC = 0.5  # a zone counts as clear when ≥50 % of its water pixels are valid
MIN_PIXELS = 20

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 150,
    }
)


def _scenes(zone_df: pd.DataFrame) -> pd.DataFrame:
    """One row per granule × zone with a `clear` flag (enough valid water pixels)."""
    s = zone_df.drop_duplicates(["granule", "zone"])[
        ["time", "granule", "zone", "n_water", "valid_frac"]
    ].copy()
    s["time"] = pd.to_datetime(s["time"], utc=True)
    s["clear"] = (s["valid_frac"] >= CLEAR_FRAC) & (s["n_water"] >= MIN_PIXELS)
    return s


def fig_availability(
    zone_df: pd.DataFrame, out: str | Path, zones=("tokyo_inner", "sagami_open")
) -> pd.DataFrame:
    """Figure 1: number of clear scenes per month for the inner Tokyo Bay and open Sagami zones."""
    s = _scenes(zone_df)
    s = s[s["zone"].isin(zones)]
    s["month"] = s["time"].dt.tz_convert("Asia/Tokyo").dt.tz_localize(None).dt.to_period("M")
    tab = s[s["clear"]].groupby(["month", "zone"]).size().unstack(fill_value=0)
    allm = pd.period_range(s["month"].min(), s["month"].max(), freq="M")
    tab = tab.reindex(allm, fill_value=0).reindex(columns=list(zones), fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = np.arange(len(tab))
    w = 0.8 / len(zones)
    for i, z in enumerate(zones):
        label, color = ZONE_STYLE[z]
        ax.bar(
            x + (i - (len(zones) - 1) / 2) * w,
            tab[z].values,
            w,
            label=f"{label} (total {tab[z].sum()})",
            color=color,
        )
    step = max(1, len(tab) // 15)
    ax.set_xticks(x[::step], [str(p) for p in tab.index[::step]], rotation=45, ha="right")
    ax.set_ylabel("Clear scenes per month")
    ax.set_title(f"PACE OCI scenes with ≥{int(CLEAR_FRAC*100)} % of zone water pixels valid", loc="left")
    ax.legend(frameon=False, ncol=len(zones))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return tab


def fig_spectra(
    zone_df: pd.DataFrame,
    out: str | Path,
    wl_range=(400, 720),
    zones=("tokyo_inner", "miura_transition", "sagami_open"),
) -> pd.DataFrame:
    """Figure 2: median reflectance spectrum per zone with its spread, clear scenes only."""
    s = _scenes(zone_df)
    clear = s[s["clear"]][["granule", "zone"]]
    d = zone_df.merge(clear, on=["granule", "zone"])
    d = d[d["zone"].isin(zones) & d["wavelength"].between(*wl_range)]
    agg = (
        d.groupby(["zone", "wavelength"])["rrs_median"]
        .agg(median="median", p25=lambda v: v.quantile(0.25), p75=lambda v: v.quantile(0.75), n="size")
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for z in zones:
        g = agg[agg["zone"] == z]
        if g.empty:
            continue
        label, color = ZONE_STYLE[z]
        n = int(g["n"].max())
        axes[0].plot(g["wavelength"], g["median"], color=color, lw=1.8, label=f"{label} (n={n})")
        axes[0].fill_between(g["wavelength"], g["p25"], g["p75"], color=color, alpha=0.15, lw=0)
        area = (
            np.trapezoid(g["median"], g["wavelength"])
            if hasattr(np, "trapezoid")
            else np.trapz(g["median"], g["wavelength"])
        )
        axes[1].plot(g["wavelength"], g["median"] / area, color=color, lw=1.8, label=label)
    axes[0].set_ylabel("Rrs (sr⁻¹)")
    axes[0].set_title("Median Rrs on clear days (band: IQR between scenes)", loc="left")
    axes[1].set_ylabel("Rrs / ∫Rrs dλ (nm⁻¹)")
    axes[1].set_title("Spectral shape (area-normalised)", loc="left")
    for ax in axes:
        ax.set_xlabel("Wavelength (nm)")
        ax.set_xlim(*wl_range)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return agg


def fig_station_chl(station_df: pd.DataFrame, out: str | Path) -> pd.DataFrame:
    """Figure 3: standard NASA chlorophyll at the pixel of each monitoring station over time."""
    d = station_df[station_df["passed"].astype(bool)].drop_duplicates(["granule", "station"]).copy()
    d["time"] = pd.to_datetime(d["time"], utc=True).dt.tz_convert("Asia/Tokyo")
    d = d.dropna(subset=["chlor_a"])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for stn, g in d.groupby("station"):
        ax.plot(g["time"], g["chlor_a"], "o-", ms=3, lw=0.8, label=f"{stn} (n={len(g)})")
    ax.set_yscale("log")
    ax.set_ylabel("chlor_a (mg m⁻³), NASA standard")
    ax.set_title("PACE standard chlorophyll at monitoring posts (3×3 window, passed QC)", loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return d


def make_all(pace_dir="data/processed/pace", fig_dir="figures") -> list[Path]:
    """Draw every figure for which the input table exists; returns the written file paths."""
    pace_dir, fig_dir = Path(pace_dir), Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []
    zf = pace_dir / "zone_spectra.csv"
    if zf.exists():
        z = pd.read_csv(zf)
        fig_availability(z, fig_dir / "01_pace_availability.png").to_csv(fig_dir / "01_pace_availability.csv")
        fig_spectra(z, fig_dir / "02_pace_spectra.png").to_csv(fig_dir / "02_pace_spectra.csv", index=False)
        made += [fig_dir / "01_pace_availability.png", fig_dir / "02_pace_spectra.png"]
    sf = pace_dir / "station_values_bgc.csv"
    if sf.exists():
        fig_station_chl(pd.read_csv(sf), fig_dir / "03_pace_station_chl.png").to_csv(
            fig_dir / "03_pace_station_chl.csv", index=False
        )
        made.append(fig_dir / "03_pace_station_chl.png")
    return made
