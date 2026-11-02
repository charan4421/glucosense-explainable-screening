"""Calibration across race/ethnicity: candidate fixes and the pre-specified decision rule
(docs/analysis_plan.md, deviation 2026-11-02 A). Exploratory; development data only.

- RaceTerms: race_eth one-hot (reference non-Hispanic White), optionally x BMI and x waist, in
  place of the single ordinal race code.
- group_recalibration: per-group intercept offsets (slope fixed at 1) estimated from inner
  cross-validated predictions on the outer training fold, applied to the outer test fold.
- decide: the adoption rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from scipy.optimize import brentq
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import cross_val_predict

from src.evaluation.metrics import _logit
from src.preprocessing.pipeline import build_model_pipeline

RACE_CODES = ["1", "2", "3", "4", "6", "7"]
REFERENCE = "3"                                   # non-Hispanic White
RELIABLE_GROUPS = ["Mexican American", "Other Hispanic", "Non-Hispanic White",
                   "Non-Hispanic Black", "Non-Hispanic Asian"]       # >= 30 positives


class RaceTerms(BaseEstimator, TransformerMixin):
    """Replaces race_eth by one-hot dummies (and dummy x BMI, dummy x waist). Stateless: the
    category list is fixed, so it cannot leak."""

    def __init__(self, interactions: bool = True):
        self.interactions = interactions

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        race = X.pop("race_eth").astype(str)
        for code in RACE_CODES:
            if code == REFERENCE:
                continue
            d = (race == code).astype(float)
            X[f"race_{code}"] = d
            if self.interactions:
                X[f"race_{code}_x_bmi"] = d * X["bmi"]
                X[f"race_{code}_x_waist"] = d * X["waist_cm"]
        return X


def race_pipeline(categorical: list[str], interactions: bool, knn_neighbors: int,
                  clinical_interactions: bool, seed: int):
    """make_pipeline hook for nested_cv: RaceTerms first, then the standard pipeline."""
    cat = [c for c in categorical if c != "race_eth"]

    def make(X_train: pd.DataFrame, model):
        terms = RaceTerms(interactions)
        pipe = build_model_pipeline(terms.transform(X_train), model, cat, "none", knn_neighbors,
                                    clinical_interactions, seed)
        out = Pipeline([("race", terms), *pipe.steps])
        out.feature_names_ = pipe.feature_names_
        return out

    return make


def group_offsets(groups, y, p, min_pos: int = 30) -> dict:
    """Intercept offset per group (logit p as offset, slope 1): solves sum(y - sigmoid(a + lp)) = 0
    within the group. Groups with fewer than min_pos positives get 0."""
    g, y, lp = np.asarray(groups, dtype=object), np.asarray(y, dtype=float), _logit(p)
    out = {}
    for level in pd.unique(g):
        m = g == level
        if y[m].sum() < min_pos:
            out[level] = 0.0
            continue
        yy, ll = y[m], lp[m]
        out[level] = float(brentq(lambda a, yy=yy, ll=ll: np.sum(yy - 1 / (1 + np.exp(-(a + ll)))),
                                  -20, 20))
    return out


def apply_offsets(groups, p, offsets: dict) -> np.ndarray:
    a = np.array([offsets.get(level, 0.0) for level in np.asarray(groups, dtype=object)])
    return 1 / (1 + np.exp(-(_logit(p) + a)))


def group_recalibration(group_col: str = "race_eth", min_pos: int = 30, log: list | None = None):
    """postprocess hook for nested_cv. Offsets come from inner cross-validated predictions of the
    tuned pipeline on the outer training fold only; `log` collects them per fold."""

    def post(search, X_train, y_train, X_test, p, inner_cv):
        p_tr = cross_val_predict(clone(search.best_estimator_), X_train, y_train, cv=inner_cv,
                                 method="predict_proba")[:, 1]
        off = group_offsets(X_train[group_col], y_train, p_tr, min_pos)
        if log is not None:
            log.append(off)
        return apply_offsets(X_test[group_col], p, off)

    return post


def oe_covers_one(sub: pd.DataFrame, groups=RELIABLE_GROUPS) -> list[str]:
    """Groups (of `groups`) whose O/E bootstrap 95 % CI includes 1."""
    r = sub[(sub.variable == "race_eth") & sub.group.isin(groups)]
    return sorted(r[(r.observed_expected_ci_low <= 1) & (r.observed_expected_ci_high >= 1)].group)


def decide(primary_sub: pd.DataFrame, primary_pr_auc: tuple[float, float],
           candidates: dict[str, tuple[pd.DataFrame, float]]) -> dict:
    """Pre-specified rule. primary_pr_auc = (mean, SD) over folds; candidates maps name ->
    (subgroup table with bootstrap CIs, mean PR-AUC over folds)."""
    base = oe_covers_one(primary_sub)
    floor = primary_pr_auc[0] - primary_pr_auc[1]
    rows = {}
    for name, (sub, pr) in candidates.items():
        covered = oe_covers_one(sub)
        rows[name] = {"groups_covered": covered, "n_covered": len(covered), "pr_auc_mean": pr,
                      "more_groups": len(covered) > len(base), "pr_auc_ok": pr >= floor}
        rows[name]["qualifies"] = rows[name]["more_groups"] and rows[name]["pr_auc_ok"]
    ok = [n for n, r in rows.items() if r["qualifies"]]
    adopted = max(ok, key=lambda n: (rows[n]["n_covered"], rows[n]["pr_auc_mean"])) if ok else None
    return {"primary_groups_covered": base, "primary_n_covered": len(base),
            "pr_auc_floor": floor, "candidates": rows, "adopted": adopted,
            "result": f"adopt {adopted} (exploratory extension)" if adopted else "negative result"}
