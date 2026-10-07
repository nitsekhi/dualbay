"""Optical water types (OWT) from Rrs spectra.

Spectra are area-normalised (shape only, magnitude removed) and clustered with k-means.
k is chosen by silhouette score within [k_min, k_max]. A fitted scheme can classify new
spectra and gives a membership distance, which is used in dualbay.transfer to flag pixels
whose optical type was absent from the training area.

Starting point for the gradient scheme should be the lab's 2020 optical classification of
Tokyo Bay; this module is the generic machinery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .baseline import band_columns


def normalise(df: pd.DataFrame, wl_range=(400, 720)) -> tuple[np.ndarray, list[str]]:
    """Area-normalised spectra within wl_range. Returns (matrix, column names)."""
    cols = [c for w, c in band_columns(df).items() if wl_range[0] <= w <= wl_range[1]]
    if len(cols) < 4:
        raise ValueError("need at least 4 bands in range")
    x = df[cols].to_numpy(dtype=float)
    x = np.where(x < 0, 0, x)                       # negative Rrs carry no shape information
    area = np.trapezoid(x, [float(c[4:]) for c in cols], axis=1) if hasattr(np, "trapezoid") else np.trapz(x, [float(c[4:]) for c in cols], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        xn = x / area[:, None]
    return xn, cols


class OWTScheme:
    def __init__(self, k_min: int = 3, k_max: int = 7, seed: int = 0):
        self.k_min, self.k_max, self.seed = k_min, k_max, seed
        self.model: KMeans | None = None
        self.cols: list[str] = []
        self.silhouettes: dict[int, float] = {}

    def fit(self, df: pd.DataFrame) -> "OWTScheme":
        xn, cols = normalise(df)
        ok = np.isfinite(xn).all(axis=1)
        xn = xn[ok]
        if len(xn) < self.k_max * 5:
            raise ValueError(f"too few spectra ({len(xn)}) for k up to {self.k_max}")
        best = None
        for k in range(self.k_min, self.k_max + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit(xn)
            s = silhouette_score(xn, km.labels_)
            self.silhouettes[k] = float(s)
            if best is None or s > best[0]:
                best = (s, km)
        self.model, self.cols = best[1], cols
        return self

    @property
    def k(self) -> int:
        return int(self.model.n_clusters)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """owt label (0..k-1), distance to the assigned centre, and whether the pixel is 'novel'
        (distance above the 95th percentile of training distances for that cluster)."""
        xn, _ = normalise(df[self.cols] if set(self.cols) <= set(df.columns) else df)
        ok = np.isfinite(xn).all(axis=1)
        lab = np.full(len(xn), -1)
        dist = np.full(len(xn), np.nan)
        if ok.any():
            lab[ok] = self.model.predict(xn[ok])
            d = self.model.transform(xn[ok])
            dist[ok] = d[np.arange(ok.sum()), lab[ok]]
        thr = self._thresholds()
        novel = np.array([bool(d > thr.get(l, np.inf)) if l >= 0 else True for l, d in zip(lab, dist)])
        return pd.DataFrame({"owt": lab, "owt_dist": dist, "owt_novel": novel}, index=df.index)

    def _thresholds(self) -> dict[int, float]:
        if not hasattr(self, "_thr"):
            # recompute from the centres' training inertia is not stored per cluster → approximate with
            # the cluster radius implied by inertia; refined in fit_thresholds()
            self._thr = {}
        return self._thr

    def fit_thresholds(self, df: pd.DataFrame, q: float = 0.95) -> "OWTScheme":
        """Per-cluster novelty thresholds from training spectra (call right after fit)."""
        xn, _ = normalise(df)
        ok = np.isfinite(xn).all(axis=1)
        lab = self.model.predict(xn[ok])
        d = self.model.transform(xn[ok])[np.arange(ok.sum()), lab]
        self._thr = {int(l): float(np.quantile(d[lab == l], q)) for l in np.unique(lab)}
        return self

    def describe(self) -> pd.DataFrame:
        """Centre spectra per OWT (for plotting and for naming the types)."""
        wl = [float(c[4:]) for c in self.cols]
        return pd.DataFrame(self.model.cluster_centers_.T, index=wl, columns=[f"OWT{i+1}" for i in range(self.k)])


def coverage(train_labels: np.ndarray, test_labels: np.ndarray) -> pd.DataFrame:
    """Share of test pixels per OWT and whether that OWT was seen in training. Core table for H2."""
    tr = pd.Series(train_labels).value_counts()
    te = pd.Series(test_labels).value_counts()
    out = pd.DataFrame({"n_train": tr, "n_test": te}).fillna(0).astype(int)
    out["seen_in_training"] = out["n_train"] > 0
    out["test_share"] = out["n_test"] / out["n_test"].sum()
    return out.sort_index()
