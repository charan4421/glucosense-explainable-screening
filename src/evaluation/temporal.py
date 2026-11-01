"""Temporal validation: tune on earlier NHANES cycles, score once on a later held-out cycle.

Also holds the feature-list guard: lab values and labels must never become model inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.data.nhanes import MAIN_LABEL, SENS_LABEL
from src.evaluation.metrics import summarize
from src.models.models import get_model
from src.preprocessing.pipeline import build_model_pipeline

# Raw NHANES lab columns and every label derived from them.
LAB_DERIVED = {"LBXGH", "LBXGLU", MAIN_LABEL, SENS_LABEL}


def _check(feats: list[str]) -> list[str]:
    leaked = {f for f in feats if f in LAB_DERIVED or f.upper() in LAB_DERIVED}
    if leaked:
        raise ValueError(f"lab-derived columns in the feature list: {sorted(leaked)}")
    gates = {f for f in feats if f.startswith("gate_")}
    if gates:
        raise ValueError(f"raw skip-pattern gate columns are not features: {sorted(gates)}")
    return feats


def feature_columns(cfg: dict) -> list[str]:
    """Set A feature list from cfg['features']; refuses lab-derived columns (label leakage)."""
    return _check(cfg["features"]["numeric"] + cfg["features"]["categorical"])


def feature_set(cfg: dict, name: str, base_for_d: str | None = None) -> tuple[list[str], list[str]]:
    """(all features, categorical features) of a nested set, following `extends`.

    Set D extends 'best_of_B_C', which must be resolved from results and passed as base_for_d."""
    sets = cfg["feature_sets"]
    spec = sets[name]
    parent = spec.get("extends")
    if parent == "best_of_B_C":
        if base_for_d not in ("B", "C"):
            raise ValueError("set D needs base_for_d = 'B' or 'C' (resolved from results)")
        parent = base_for_d
    feats, cat = feature_set(cfg, parent) if parent else ([], [])
    feats = feats + [f for f in spec["numeric"] + spec["categorical"] if f not in feats]
    cat = cat + [c for c in spec["categorical"] if c not in cat]
    return _check(feats), cat


def temporal_split(df: pd.DataFrame, test_cycle: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = df[df.cycle != test_cycle], df[df.cycle == test_cycle]
    if test.empty:
        raise ValueError(f"no rows for test cycle {test_cycle}")
    later = [c for c in train.cycle.unique() if c > test_cycle]
    if later:
        raise ValueError(f"training cycles {later} are later than the test cycle {test_cycle}")
    return train, test


def temporal_evaluate(df: pd.DataFrame, feats: list[str], categorical: list[str], target: str,
                      test_cycle: str, model_name: str, strategy: str,
                      cfg: dict) -> tuple[dict, pd.DataFrame]:
    """Tune with inner CV on the training cycles only, then score once on the test cycle.

    Returns (metrics row, test predictions with columns y_true and p)."""
    train, test = temporal_split(df, test_cycle)
    seed, cv = cfg["seed"], cfg["cv"]
    model, space = get_model(model_name, seed)
    pipe = build_model_pipeline(train[feats], model, categorical, strategy,
                                cfg["preprocessing"]["knn_neighbors"],
                                cfg["features"]["interactions"], seed)
    search = RandomizedSearchCV(pipe, space, n_iter=cv["n_iter"],
                                cv=StratifiedKFold(cv["inner_folds"], shuffle=True, random_state=seed),
                                scoring=cv["scoring"], random_state=seed, n_jobs=1, refit=True)
    search.fit(train[feats], train[target])
    p = search.predict_proba(test[feats])[:, 1]
    m = summarize(test[target], p)
    row = {"model": model_name, "strategy": strategy, "test_cycle": test_cycle,
           "train_cycles": ",".join(sorted(train.cycle.unique())),
           "n_train": len(train), "n_test": len(test),
           "train_prevalence": float(np.mean(train[target])),
           "test_prevalence": m.pop("prevalence"), **m,
           "best_params": str(search.best_params_)}
    preds = pd.DataFrame({"y_true": test[target].to_numpy(), "p": p}, index=test.index)
    return row, preds
