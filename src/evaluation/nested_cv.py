"""Nested cross-validation. Outer folds estimate performance, inner folds tune hyperparameters.

Every fold fits the whole pipeline (imputation, scaling, SMOTE-NC, model) on training data only.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold

from src.evaluation.metrics import calibration_stats, summarize
from src.models.models import RAW_INPUT_MODELS, get_model
from src.preprocessing.pipeline import build_model_pipeline


def nested_cv(X: pd.DataFrame, y, model_name: str, strategy: str, categorical: list[str],
              outer_folds: int = 5, inner_folds: int = 3, n_iter: int = 15,
              scoring: str = "average_precision", knn_neighbors: int = 5,
              interactions: bool = True, seed: int = 42, outer_repeats: int = 1):
    """Returns (per-fold metrics DataFrame, out-of-fold probabilities).

    With outer_repeats > 1 the outer split is repeated with new shuffles and the out-of-fold
    probabilities are averaged over repeats. Folds depend only on (y, seed), so every model and
    feature set sees the same folds."""
    y = np.asarray(y)
    if outer_repeats > 1:
        outer = RepeatedStratifiedKFold(n_splits=outer_folds, n_repeats=outer_repeats,
                                        random_state=seed)
    else:
        outer = StratifiedKFold(outer_folds, shuffle=True, random_state=seed)
    inner = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
    oof_sum, oof_n = np.zeros(len(y)), np.zeros(len(y))
    rows = []
    for i, (tr, te) in enumerate(outer.split(X, y)):
        rep, k = divmod(i, outer_folds)
        model, space = get_model(model_name, seed)
        if model_name == "tabnet":
            strategy_here = strategy if strategy != "class_weight" else "none"  # no class_weight in wrapper
        else:
            strategy_here = strategy
        if model_name == "mlp" and strategy == "class_weight":
            model.set_params(pos_weight="auto")
        t0 = time.perf_counter()
        if model_name in RAW_INPUT_MODELS:            # fixed score: fit its mapping, no search
            search = model.fit(X.iloc[tr], y[tr])
            best_params = {}
        else:
            pipe = build_model_pipeline(X.iloc[tr], model, categorical, strategy_here,
                                        knn_neighbors, interactions, seed)
            search = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=inner, scoring=scoring,
                                        random_state=seed, n_jobs=1, refit=True)
            search.fit(X.iloc[tr], y[tr])
            best_params = search.best_params_
        p = search.predict_proba(X.iloc[te])[:, 1]
        oof_sum[te] += p
        oof_n[te] += 1
        cal = calibration_stats(y[te], p)          # on this outer fold's predictions only
        rows.append({"model": model_name, "strategy": strategy, "repeat": rep, "fold": k,
                     "n_iter": n_iter,
                     "best_params": str(best_params), **summarize(y[te], p),
                     "citl_mean_predicted": cal["citl_mean_predicted"],
                     "citl_difference": cal["citl_difference"],
                     "citl_ratio_observed_expected": cal["citl_ratio_observed_expected"],
                     "calibration_slope": cal["slope"],
                     "calibration_intercept": cal["intercept"],
                     "calibration_intercept_offset": cal["intercept_offset"],
                     "n_pos": int(y[te].sum()), "n": len(te),
                     "fit_seconds": time.perf_counter() - t0})
    return pd.DataFrame(rows), oof_sum / oof_n
