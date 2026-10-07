"""Baseline algorithms, local recalibration and accuracy metrics.

Three tiers (project-brief):
  tier 1  standard algorithms with their published coefficients
  tier 2  the same algorithm forms refitted on the project's own matchups  ← the −20 % reference
  tier 3  project models (dualbay.models)

Spectra are passed as a DataFrame with columns rrs_<nm>; the nearest available band is used,
so the same code runs on PACE (5 nm), SGLI, OLCI or in-situ hyperspectral data.

References
  OCx     O'Reilly et al. 1998, JGR 103:24937; coefficients: OC4v6 (SeaWiFS) as a generic start
  CI      Hu, Lee & Franz 2012, JGR 117:C01011
  3-band  Gitelson et al. 2008, RSE 112:3582
  NDCI    Mishra & Mishra 2012, RSE 117:394
  Nechad  Nechad, Ruddick & Park 2010, RSE 114:854 (TSM, 665 nm)
  Dogliotti et al. 2015, RSE 156:157 (turbidity, 645 / 859 nm switching)
  Metrics MdSA and bias: Morley, Brito & Welling 2018; Seegers et al. 2018
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ---------- band access ----------

def band_columns(df: pd.DataFrame) -> dict[float, str]:
    out = {}
    for c in df.columns:
        if isinstance(c, str) and c.startswith("rrs_"):
            try:
                out[float(c[4:])] = c
            except ValueError:
                pass
    return dict(sorted(out.items()))


def band(df: pd.DataFrame, wl: float, tol: float = 12.0) -> np.ndarray:
    """Rrs at the band nearest to `wl` (within `tol` nm)."""
    cols = band_columns(df)
    if not cols:
        raise ValueError("no rrs_<nm> columns")
    nearest = min(cols, key=lambda w: abs(w - wl))
    if abs(nearest - wl) > tol:
        raise ValueError(f"no band within {tol} nm of {wl} nm (nearest {nearest})")
    return df[cols[nearest]].to_numpy(dtype=float)


# ---------- algorithm forms (x = feature, returns parameter) ----------

def _ocx_feature(df):
    """log10 of max blue / green ratio (OC4 form: max(443,490,510)/555)."""
    blues = [band(df, w) for w in (443, 490, 510)]
    green = band(df, 555)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(np.nanmax(np.vstack(blues), axis=0) / green)


def ocx_poly(x, a0, a1, a2, a3, a4):
    return 10 ** (a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4)


OC4V6 = (0.3272, -2.9940, 2.7218, -1.2259, -0.5683)


def _ci_feature(df):
    """Color index: Rrs(555) - [Rrs(443) + (555-443)/(670-443) (Rrs(670)-Rrs(443))]."""
    r443, r555, r670 = band(df, 443), band(df, 555), band(df, 670)
    return r555 - (r443 + (555 - 443) / (670 - 443) * (r670 - r443))


def ci_exp(x, a, b):
    return 10 ** (a + b * x)


HU2012 = (-0.4909, 191.6590)


def _threeband_feature(df):
    """Gitelson 3-band: (1/Rrs665 - 1/Rrs708) * Rrs753."""
    r665, r708, r753 = band(df, 665), band(df, 708), band(df, 753)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (1 / r665 - 1 / r708) * r753


def _ndci_feature(df):
    r665, r708 = band(df, 665), band(df, 708)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (r708 - r665) / (r708 + r665)


def linear(x, a, b):
    return a + b * x


def quadratic(x, a, b, c):
    return a + b * x + c * x**2


def nechad(rho_w, A, C):
    """TSM = A * rho_w / (1 - rho_w / C); rho_w = pi * Rrs."""
    return A * rho_w / (1 - rho_w / C)


NECHAD_665 = (355.85, 0.1728)      # Nechad et al. 2010, Table 4 (665 nm), g m^-3
DOGLIOTTI_645 = (228.1, 0.1641)    # FNU
DOGLIOTTI_859 = (3078.9, 0.2112)


def dogliotti_turbidity(df):
    """Dogliotti 2015 with the red/NIR switch on rho_w(645)."""
    rho645, rho859 = np.pi * band(df, 645), np.pi * band(df, 859)
    t_red = nechad(rho645, *DOGLIOTTI_645)
    t_nir = nechad(rho859, *DOGLIOTTI_859)
    w = np.clip((rho645 - 0.05) / (0.07 - 0.05), 0, 1)   # blend between 0.05 and 0.07
    return (1 - w) * t_red + w * t_nir


# registry: name -> (parameter, feature fn, model fn, published coefficients, bounds for refit)
ALGORITHMS = {
    "ocx":     ("chl", _ocx_feature, ocx_poly, OC4V6, None),
    "ci":      ("chl", _ci_feature, ci_exp, HU2012, None),
    "3band":   ("chl", _threeband_feature, linear, (0.0, 1.0), None),       # no universal coefficients → tier 2 only
    "ndci":    ("chl", _ndci_feature, quadratic, (14.039, 86.115, 194.325), None),
    "nechad":  ("tsm", lambda d: np.pi * band(d, 665), nechad, NECHAD_665, ([0, 0.01], [1e5, 1.0])),
}


# ---------- metrics ----------

def metrics(y, yhat, lo=None, hi=None) -> dict:
    """Standard ocean-colour accuracy metrics. Log-space metrics are computed on positive pairs only."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    n = int(ok.sum())
    if n == 0:
        return {"n": 0}
    err = yhat - y
    pos = (y > 0) & (yhat > 0)
    out = {
        "n": n,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "mape": float(100 * np.mean(np.abs(err[y != 0] / y[y != 0]))) if np.any(y != 0) else np.nan,
        "r2": float(1 - np.sum(err**2) / np.sum((y - y.mean())**2)) if n > 1 and np.var(y) > 0 else np.nan,
    }
    if pos.sum() > 1:
        lr = np.log10(yhat[pos] / y[pos])
        out["mdsa"] = float(100 * (10 ** np.median(np.abs(lr)) - 1))   # median symmetric accuracy, %
        out["log_bias"] = float(10 ** np.median(lr))                   # multiplicative bias
    if lo is not None and hi is not None:
        lo, hi = np.asarray(lo, float)[ok], np.asarray(hi, float)[ok]
        out["picp68"] = float(np.mean((y >= lo) & (y <= hi)))           # coverage of the 68 % interval
    return out


# ---------- tiers ----------

def apply_standard(df: pd.DataFrame, name: str) -> np.ndarray:
    """Tier 1: published coefficients."""
    param, feat, fn, coef, _ = ALGORITHMS[name]
    if name == "3band":
        raise ValueError("3band has no universal coefficients; use recalibrate()")
    with np.errstate(all="ignore"):
        return fn(feat(df), *coef)


def recalibrate(df: pd.DataFrame, y: np.ndarray, name: str):
    """Tier 2: refit the algorithm form on the project's matchups. Returns (coefficients, predict fn)."""
    param, feat, fn, coef0, bounds = ALGORITHMS[name]
    x = feat(df)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if ok.sum() < len(coef0) + 3:
        raise ValueError(f"{name}: only {ok.sum()} usable matchups for {len(coef0)} coefficients")
    kw = {"bounds": bounds} if bounds else {}
    try:
        if fn in (ocx_poly, ci_exp):
            # fit in log space so large values do not dominate
            logfn = lambda xx, *p: np.log10(np.clip(fn(xx, *p), 1e-6, None))
            popt, _ = curve_fit(logfn, x[ok], np.log10(y[ok]), p0=coef0, maxfev=20000, **kw)
        else:
            popt, _ = curve_fit(fn, x[ok], y[ok], p0=coef0, maxfev=20000, **kw)
    except RuntimeError as err:
        raise RuntimeError(f"{name}: refit did not converge ({err})") from err
    return tuple(popt), (lambda d: fn(feat(d), *popt))


def run_tiers(train: pd.DataFrame, test: pd.DataFrame, param: str = "chl") -> pd.DataFrame:
    """Evaluate every algorithm of `param` at tier 1 and tier 2. Returns one row per (algorithm, tier)."""
    rows = []
    for name, (p, *_ ) in ALGORITHMS.items():
        if p != param:
            continue
        if name != "3band":
            try:
                yhat = apply_standard(test, name)
                rows.append({"algorithm": name, "tier": 1, **metrics(test["y"], yhat)})
            except ValueError as err:
                rows.append({"algorithm": name, "tier": 1, "n": 0, "note": str(err)})
        try:
            _, predict = recalibrate(train, train["y"].to_numpy(), name)
            yhat = predict(test)
            rows.append({"algorithm": name, "tier": 2, **metrics(test["y"], yhat)})
        except (ValueError, RuntimeError) as err:
            rows.append({"algorithm": name, "tier": 2, "n": 0, "note": str(err)})
    return pd.DataFrame(rows)
