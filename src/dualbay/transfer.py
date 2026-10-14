"""Transfer experiments (RQ2) and the 'adaptive vs static' test.

Inputs: an analysis table from db.matchup_spectra() — one row per matchup with columns
  y, bay, zone_id, scene_time, rrs_<nm>  (+ owt after OWT assignment)

Experiments
  temporal_holdout    train on years A, test on year B; per-season breakdown
  static_vs_adaptive  for each month of the test year: static model (frozen) vs a model
                      refitted with everything observed up to the previous month
  spatial_transfer    train on one bay, test on the other; coverage by OWT explains failures
  learning_curve      how many target-bay matchups are needed to reach a target metric
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .baseline import metrics
from .models import RFRetrieval
from .owt import OWTScheme, coverage

SEASONS = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}


def _season(ts: pd.Series) -> pd.Series:
    """Month → season label (DJF, MAM, JJA, SON)."""
    return ts.dt.month.map(SEASONS)


def _fit(train: pd.DataFrame, seed=0) -> RFRetrieval:
    """Fit the project model on a training table."""
    return RFRetrieval(seed=seed).fit(train, train["y"])


def _eval_subsets(model: RFRetrieval, test: pd.DataFrame, by: str | None) -> pd.DataFrame:
    """Metrics for the whole test set and, if `by` is a column, for each of its groups."""
    p = model.predict(test)
    rows = [{"subset": "all", **metrics(test["y"], p["y_pred"], p["y_lo"], p["y_hi"])}]
    if by and by in test:
        for k, idx in test.groupby(by).groups.items():
            rows.append(
                {
                    "subset": f"{by}:{k}",
                    **metrics(
                        test.loc[idx, "y"], p.loc[idx, "y_pred"], p.loc[idx, "y_lo"], p.loc[idx, "y_hi"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def temporal_holdout(df: pd.DataFrame, train_years, test_year: int) -> pd.DataFrame:
    d = df.copy()
    d["year"] = d["scene_time"].dt.year
    d["season"] = _season(d["scene_time"])
    train, test = d[d["year"].isin(train_years)], d[d["year"] == test_year]
    if len(train) < 30 or len(test) < 10:
        raise ValueError(f"not enough matchups: train={len(train)}, test={len(test)}")
    return _eval_subsets(_fit(train), test, "season").assign(
        experiment=f"train {sorted(train_years)} → test {test_year}"
    )


def static_vs_adaptive(df: pd.DataFrame, train_years, test_year: int, min_new: int = 5) -> pd.DataFrame:
    """Monthly comparison in the test year.

    static   model fitted once on train_years
    adaptive model refitted at the start of each month on train_years + all test-year matchups before that month
    """
    d = df.copy()
    d["year"] = d["scene_time"].dt.year
    d["month"] = d["scene_time"].dt.month
    base = d[d["year"].isin(train_years)]
    test = d[d["year"] == test_year]
    if len(base) < 30:
        raise ValueError("not enough training matchups")
    static = _fit(base)
    rows = []
    for m in sorted(test["month"].unique()):
        this = test[test["month"] == m]
        seen = test[test["month"] < m]
        ps = static.predict(this)
        adaptive = _fit(pd.concat([base, seen])) if len(seen) >= min_new else static
        pa = adaptive.predict(this)
        ms, ma = metrics(this["y"], ps["y_pred"]), metrics(this["y"], pa["y_pred"])
        rows.append(
            {
                "month": int(m),
                "n": len(this),
                "n_new_used": len(seen),
                "rmse_static": ms.get("rmse"),
                "rmse_adaptive": ma.get("rmse"),
                "mdsa_static": ms.get("mdsa"),
                "mdsa_adaptive": ma.get("mdsa"),
            }
        )
    out = pd.DataFrame(rows)
    out["gain_pct"] = 100 * (out["rmse_static"] - out["rmse_adaptive"]) / out["rmse_static"]
    return out


def spatial_transfer(
    df: pd.DataFrame, source_bay: str, target_bay: str, scheme: OWTScheme | None = None
) -> dict:
    """Train on source bay, test on target bay. Returns metrics and the OWT coverage table."""
    src, tgt = df[df["bay"] == source_bay].copy(), df[df["bay"] == target_bay].copy()
    if len(src) < 30 or len(tgt) < 10:
        raise ValueError(f"not enough matchups: {source_bay}={len(src)}, {target_bay}={len(tgt)}")
    scheme = scheme or OWTScheme().fit(src).fit_thresholds(src)
    src["owt"] = scheme.predict(src)["owt"]
    tp = scheme.predict(tgt)
    seen = set(src["owt"].unique())
    # novel = far from its centre OR an optical type that never occurs in the source bay
    tgt["owt"], tgt["owt_novel"] = tp["owt"], tp["owt_novel"] | ~tp["owt"].isin(seen)
    model = _fit(src)
    p = model.predict(tgt)
    res = {
        "experiment": f"{source_bay} → {target_bay}",
        "metrics_all": metrics(tgt["y"], p["y_pred"], p["y_lo"], p["y_hi"]),
        "metrics_seen_owt": metrics(tgt.loc[~tgt["owt_novel"], "y"], p.loc[~tgt["owt_novel"], "y_pred"]),
        "metrics_novel_owt": metrics(tgt.loc[tgt["owt_novel"], "y"], p.loc[tgt["owt_novel"], "y_pred"]),
        "coverage": coverage(src["owt"].to_numpy(), tgt["owt"].to_numpy()),
        "share_novel": float(tgt["owt_novel"].mean()),
    }
    return res


def learning_curve(
    df: pd.DataFrame,
    source_bay: str,
    target_bay: str,
    sizes=(0, 5, 10, 20, 40, 80),
    repeats: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Fine-tune with n target-bay matchups (n in sizes) and evaluate on the remaining target data.
    Answers 'how many local samples does a new bay need'."""
    rng = np.random.default_rng(seed)
    src, tgt = df[df["bay"] == source_bay], df[df["bay"] == target_bay]
    rows = []
    for n in sizes:
        if n >= len(tgt) - 10:
            break
        for r in range(repeats):
            idx = rng.permutation(len(tgt))
            add, hold = tgt.iloc[idx[:n]], tgt.iloc[idx[n:]]
            model = _fit(pd.concat([src, add]), seed=r)
            p = model.predict(hold)
            rows.append({"n_local": n, "repeat": r, **metrics(hold["y"], p["y_pred"])})
    out = pd.DataFrame(rows)
    return out.groupby("n_local")[["rmse", "mdsa", "r2"]].agg(["mean", "std"]).reset_index()
