"""Project retrieval models (tier 3) with per-pixel uncertainty.

Model: random forest on log10(Rrs) features + OWT as a categorical feature, trained in
log10(parameter) space. Uncertainty: the spread of the per-tree predictions gives the central
68 % interval (quantiles of the tree ensemble). This is the simplest method that
(a) produces an interval per pixel, (b) can be checked by coverage (PICP), and
(c) runs on a laptop with a few hundred matchups. GP or MDN can replace it behind the same API.

Feature engineering keeps spectra comparable across sensors: bands are resampled to a common
grid by nearest-neighbour, so a model trained on PACE can be applied to SGLI bands at a loss of
spectral detail (used only in cross-sensor checks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .baseline import band_columns, metrics

COMMON_GRID = list(range(400, 721, 10))  # nm


class RFRetrieval:
    def __init__(self, n_estimators: int = 400, min_samples_leaf: int = 3, use_owt: bool = True, seed: int = 0):
        self.rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                        random_state=seed, n_jobs=-1)
        self.use_owt = use_owt
        self.feature_names: list[str] = []

    # ---- features
    def _features(self, df: pd.DataFrame) -> np.ndarray:
        cols = band_columns(df)
        if not cols:
            raise ValueError("no rrs_<nm> columns")
        wls = np.array(list(cols))
        feats = []
        for g in COMMON_GRID:
            nearest = wls[np.argmin(np.abs(wls - g))]
            if abs(nearest - g) <= 6:
                feats.append(np.log10(np.clip(df[cols[nearest]].to_numpy(float), 1e-5, None)))
            else:
                feats.append(np.full(len(df), np.nan))
        x = np.vstack(feats).T
        # band ratios that carry most of the Chl-a signal; cheap and robust
        def b(w):
            nearest = wls[np.argmin(np.abs(wls - w))]
            return df[cols[nearest]].to_numpy(float)
        with np.errstate(all="ignore"):
            x = np.column_stack([x, np.log10(np.clip(b(490) / b(555), 1e-3, None)),
                                 np.log10(np.clip(b(665) / b(555), 1e-3, None)),
                                 (b(708) - b(665)) / (b(708) + b(665) + 1e-9)])
        names = [f"logrrs_{g}" for g in COMMON_GRID] + ["log_490_555", "log_665_555", "ndci"]
        if self.use_owt and "owt" in df:
            x = np.column_stack([x, df["owt"].to_numpy(float)])
            names.append("owt")
        self.feature_names = names
        # RF cannot take NaN: fill with column medians computed on the fly
        med = np.nanmedian(x, axis=0)
        inds = np.where(np.isnan(x))
        x[inds] = np.take(med, inds[1])
        return x

    def fit(self, df: pd.DataFrame, y) -> "RFRetrieval":
        y = np.asarray(y, float)
        ok = np.isfinite(y) & (y > 0)
        self.rf.fit(self._features(df[ok]), np.log10(y[ok]))
        return self

    def predict(self, df: pd.DataFrame, q=(0.16, 0.84)) -> pd.DataFrame:
        """y_pred (median of trees), y_lo / y_hi (central 68 %), and the ensemble spread as 'y_sd_log'."""
        x = self._features(df)
        trees = np.vstack([t.predict(x) for t in self.rf.estimators_])   # (n_trees, n)
        med = np.median(trees, axis=0)
        lo, hi = np.quantile(trees, q[0], axis=0), np.quantile(trees, q[1], axis=0)
        return pd.DataFrame({"y_pred": 10**med, "y_lo": 10**lo, "y_hi": 10**hi, "y_sd_log": trees.std(axis=0)}, index=df.index)

    def importance(self) -> pd.Series:
        return pd.Series(self.rf.feature_importances_, index=self.feature_names).sort_values(ascending=False)


def evaluate(model: RFRetrieval, test: pd.DataFrame) -> dict:
    p = model.predict(test)
    return metrics(test["y"], p["y_pred"], p["y_lo"], p["y_hi"])


def calibration_table(y, pred: pd.DataFrame, levels=(0.5, 0.68, 0.9)) -> pd.DataFrame:
    """Nominal vs observed interval coverage. Needs the model to be re-queried per level, so this
    works from the log-space sd assuming a normal tree distribution — adequate for a first check."""
    from scipy.stats import norm

    y = np.asarray(y, float)
    mu = np.log10(pred["y_pred"].to_numpy())
    sd = pred["y_sd_log"].to_numpy()
    rows = []
    for lv in levels:
        z = norm.ppf(0.5 + lv / 2)
        inside = (np.log10(y) >= mu - z * sd) & (np.log10(y) <= mu + z * sd)
        rows.append({"nominal": lv, "observed": float(np.mean(inside)), "n": int(len(y))})
    return pd.DataFrame(rows)
