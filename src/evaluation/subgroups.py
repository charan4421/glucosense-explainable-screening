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


def subgroup_table(y, p, groups: dict[str, pd.Series], min_pos: int = 30) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    rows = []

    def row(variable, group, m):
        yy, pp = y[m], p[m]
        n_pos = int(yy.sum())
        r = {"variable": variable, "group": group, "n": int(m.sum()), "n_pos": n_pos,
             "prevalence": yy.mean(), "reliable": n_pos >= min_pos,
             "note": "" if n_pos >= min_pos else f"fewer than {min_pos} positives: not reliable"}
        if 0 < n_pos < len(yy):                       # metrics need both classes
            s, c = summarize(yy, pp), calibration_stats(yy, pp)
            r.update(pr_auc=s["pr_auc"], roc_auc=s["roc_auc"], calibration_slope=c["slope"],
                     observed_expected=c["citl_ratio_observed_expected"])
        return r

    rows.append(row("overall", "all", np.ones(len(y), dtype=bool)))
    for variable, g in groups.items():
        g = np.asarray(g, dtype=object)
        for level in pd.unique(g[pd.notna(g)]):
            rows.append(row(variable, level, g == level))
    return pd.DataFrame(rows)
