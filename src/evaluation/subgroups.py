"""Subgroup performance on pooled out-of-fold predictions.

Groups with fewer than `min_pos` positives are flagged unreliable: their PR-AUC, slope and O/E
rest on a handful of cases and should not be read as estimates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import calibration_stats, summarize

RACE_ETH = {"1": "Mexican American", "2": "Other Hispanic", "3": "Non-Hispanic White",
            "4": "Non-Hispanic Black", "6": "Non-Hispanic Asian", "7": "Other / multiracial"}


BOOT_METRICS = ["pr_auc", "roc_auc", "calibration_slope", "observed_expected"]


def _metrics(y, p) -> dict:
    s, c = summarize(y, p), calibration_stats(y, p)
    return {"pr_auc": s["pr_auc"], "roc_auc": s["roc_auc"], "calibration_slope": c["slope"],
            "observed_expected": c["citl_ratio_observed_expected"]}


def _bootstrap(y, p, n_boot: int, rng, alpha: float = 0.05) -> dict:
    """Percentile intervals from resampling the group's own rows; one-class resamples skipped."""
    draws = {k: [] for k in BOOT_METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        for k, v in _metrics(y[idx], p[idx]).items():
            draws[k].append(v)
    out = {"n_boot_used": len(draws["pr_auc"])}
    for k, v in draws.items():
        lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if v else (np.nan,) * 2
        out[f"{k}_ci_low"], out[f"{k}_ci_high"] = lo, hi
    return out


def subgroup_table(y, p, groups: dict[str, pd.Series], min_pos: int = 30, n_boot: int = 0,
                   seed: int = 42) -> pd.DataFrame:
    """One row per group; with n_boot > 0, adds percentile 95 % intervals resampled within it."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    rows = []

    def row(variable, group, m):
        yy, pp = y[m], p[m]
        n_pos = int(yy.sum())
        r = {"variable": variable, "group": group, "n": int(m.sum()), "n_pos": n_pos,
             "prevalence": yy.mean(), "reliable": n_pos >= min_pos,
             "note": "" if n_pos >= min_pos else f"fewer than {min_pos} positives: not reliable"}
        if 0 < n_pos < len(yy):                       # metrics need both classes
            r.update(_metrics(yy, pp))
            if n_boot:
                r.update(_bootstrap(yy, pp, n_boot, rng))
        return r

    rows.append(row("overall", "all", np.ones(len(y), dtype=bool)))
    for variable, g in groups.items():
        g = np.asarray(g, dtype=object)
        for level in pd.unique(g[pd.notna(g)]):
            rows.append(row(variable, level, g == level))
    return pd.DataFrame(rows)
