"""Nested cross-validation. Outer folds estimate performance, inner folds tune hyperparameters.

Every fold fits the whole pipeline (imputation, scaling, SMOTE-NC, model) on training data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.evaluation.metrics import summarize
from src.models.models import get_model
from src.preprocessing.pipeline import build_model_pipeline


def nested_cv(X: pd.DataFrame, y, model_name: str, strategy: str, categorical: list[str],
              outer_folds: int = 5, inner_folds: int = 3, n_iter: int = 15,
              scoring: str = "average_precision", knn_neighbors: int = 5,
              interactions: bool = True, seed: int = 42):
    """Returns (per-fold metrics DataFrame, out-of-fold probabilities)."""
    y = np.asarray(y)
    outer = StratifiedKFold(outer_folds, shuffle=True, random_state=seed)
    inner = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    rows = []
    for k, (tr, te) in enumerate(outer.split(X, y)):
        model, space = get_model(model_name, seed)
        if model_name == "tabnet":
            strategy_here = strategy if strategy != "class_weight" else "none"  # no class_weight in wrapper
        else:
            strategy_here = strategy
        if model_name == "mlp" and strategy == "class_weight":
            model.set_params(pos_weight="auto")
        pipe = build_model_pipeline(X.iloc[tr], model, categorical, strategy_here,
                                    knn_neighbors, interactions, seed)
        search = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=inner, scoring=scoring,
                                    random_state=seed, n_jobs=1, refit=True)
        search.fit(X.iloc[tr], y[tr])
        p = search.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        rows.append({"model": model_name, "strategy": strategy, "fold": k,
                     "best_params": str(search.best_params_), **summarize(y[te], p)})
    return pd.DataFrame(rows), oof
