"""Pre-specified selection rules (docs/analysis_plan.md), applied to the nested-CV summary.

- Primary set: for each set, its best main model by mean PR-AUC. The overall best is the highest
  of those; the primary set is the smallest set (A < B < C) whose best mean PR-AUC is within one
  SD (across outer folds) of the overall best.
- Primary model: on the primary set, the simplest model (simplicity order from the config) whose
  mean PR-AUC is within one SD of the best model on that set.
- Base of set D: whichever of B and C has the higher mean PR-AUC averaged over the main models.
"""
from __future__ import annotations

import pandas as pd

SET_ORDER = ["A", "B", "C"]


def _main(summary: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    return summary[(summary.strategy == "none") & summary.model.isin(models)
                   & summary.feature_set.isin(SET_ORDER)]


def primary_set(summary: pd.DataFrame, models: list[str]) -> dict:
    m = _main(summary, models)
    best_per_set = m.loc[m.groupby("feature_set").pr_auc_mean.idxmax()].set_index("feature_set")
    top = best_per_set.pr_auc_mean.idxmax()
    threshold = best_per_set.loc[top, "pr_auc_mean"] - best_per_set.loc[top, "pr_auc_sd"]
    chosen = next(s for s in SET_ORDER if s in best_per_set.index
                  and best_per_set.loc[s, "pr_auc_mean"] >= threshold)
    return {"primary_set": chosen, "best_set": top, "best_model": best_per_set.loc[top, "model"],
            "best_pr_auc": best_per_set.loc[top, "pr_auc_mean"],
            "best_pr_auc_sd": best_per_set.loc[top, "pr_auc_sd"], "threshold": threshold}


def primary_model(summary: pd.DataFrame, feature_set: str, simplicity_order: list[str]) -> dict:
    m = _main(summary, simplicity_order)
    m = m[m.feature_set == feature_set].set_index("model")
    top = m.pr_auc_mean.idxmax()
    threshold = m.loc[top, "pr_auc_mean"] - m.loc[top, "pr_auc_sd"]
    chosen = next(x for x in simplicity_order if x in m.index and m.loc[x, "pr_auc_mean"] >= threshold)
    return {"primary_model": chosen, "best_model": top, "best_pr_auc": m.loc[top, "pr_auc_mean"],
            "best_pr_auc_sd": m.loc[top, "pr_auc_sd"], "threshold": threshold}


def base_for_d(summary: pd.DataFrame, models: list[str]) -> str:
    m = _main(summary, models)
    means = m[m.feature_set.isin(["B", "C"])].groupby("feature_set").pr_auc_mean.mean()
    if set(means.index) != {"B", "C"}:
        raise ValueError("sets B and C must both be finished before set D")
    return means.idxmax()
