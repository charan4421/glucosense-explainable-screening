"""Paired comparisons on identical outer folds, with the Nadeau-Bengio corrected interval.

Fold scores from (repeated) k-fold CV are not independent: training sets overlap, so the naive
SD of fold differences understates the uncertainty. Nadeau & Bengio (2003) inflate the variance:

    var = (1/J + n_test/n_train) * s^2,   J = number of folds (k x repeats), s^2 = sample variance

and the 95 % interval is mean +/- t(J-1, 0.975) * sqrt(var). For k-fold, n_test/n_train = 1/(k-1).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

KEY = ["repeat", "fold"]


def nadeau_bengio(diff: np.ndarray, test_train_ratio: float, alpha: float = 0.05) -> dict:
    diff = np.asarray(diff, dtype=float)
    j = len(diff)
    mean, s2 = diff.mean(), diff.var(ddof=1)
    se = np.sqrt((1 / j + test_train_ratio) * s2)
    half = stats.t.ppf(1 - alpha / 2, j - 1) * se
    t = mean / se if se > 0 else np.nan
    return {"mean_diff": mean, "ci_low": mean - half, "ci_high": mean + half, "n_folds": j,
            "p_value": 2 * stats.t.sf(abs(t), j - 1) if se > 0 else np.nan}


def paired(folds: pd.DataFrame, a: dict, b: dict, metric: str, k: int) -> dict:
    """Difference a - b on the folds both share; a and b select rows (e.g. model, feature_set)."""
    def pick(sel):
        m = np.logical_and.reduce([folds[c] == v for c, v in sel.items()])
        return folds[m].set_index(KEY)
    fa, fb = pick(a), pick(b)
    common = fa.index.intersection(fb.index)
    if len(common) < 2:
        raise ValueError(f"fewer than 2 shared folds for {a} vs {b}")
    if not (fa.loc[common, "n"] == fb.loc[common, "n"]).all():
        raise ValueError(f"folds differ in size for {a} vs {b}: not identical folds")
    res = nadeau_bengio(fa.loc[common, metric] - fb.loc[common, metric], 1 / (k - 1))
    return {"metric": metric, **{f"a_{c}": v for c, v in a.items()},
            **{f"b_{c}": v for c, v in b.items()}, **res}


def comparison_table(folds: pd.DataFrame, models: list[str], sets: list[str], k: int,
                     metrics=("pr_auc", "roc_auc", "brier")) -> pd.DataFrame:
    """All model pairs within each set, and every set pair within each model (strategy none)."""
    f = folds[folds.strategy == "none"]
    rows = []
    for metric in metrics:
        for s in sets:
            have = [m for m in models if ((f.model == m) & (f.feature_set == s)).any()]
            for m1, m2 in combinations(have, 2):
                rows.append({"comparison": "model", **paired(
                    f, {"model": m1, "feature_set": s}, {"model": m2, "feature_set": s}, metric, k)})
        for m in models:
            have = [s for s in sets if ((f.model == m) & (f.feature_set == s)).any()]
            for s1, s2 in combinations(have, 2):
                rows.append({"comparison": "set", **paired(
                    f, {"model": m, "feature_set": s2}, {"model": m, "feature_set": s1}, metric, k)})
    return pd.DataFrame(rows)
