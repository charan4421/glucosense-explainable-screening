"""Portable copy of a fitted one-hot logistic-regression pipeline, as plain JSON.

The saved sklearn model cannot be shared as is: its KNN imputer stores every training row
(NHANES participants). The export keeps only aggregate parameters (scaler medians/IQRs, the
most frequent category, one-hot levels, coefficients), so it needs complete numeric inputs;
predictions on complete rows equal the sklearn pipeline's (checked in scripts/export_demo_model.py
and tests/test_portable.py). Scoring needs only numpy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.preprocessing.pipeline import ClinicalFeatures

INPUTS = ["age", "sex", "bmi", "waist_cm", "height_cm", "systolic_bp", "diastolic_bp",
          "family_history_diabetes", "smoker", "race_eth"]


def export(pipe) -> dict:
    """pipe: fitted Pipeline (or a search whose best_estimator_ is one) built by
    build_model_pipeline with interactions and one-hot encoding, ending in LogisticRegression."""
    pipe = getattr(pipe, "best_estimator_", pipe)
    ct, lr = pipe.named_steps["columns"], pipe.named_steps["model"]
    (_, num_pipe, num), (_, cat_pipe, cat) = ct.transformers_[:2]
    scale, imp, enc = num_pipe.named_steps["scale"], cat_pipe.named_steps["impute"], \
        cat_pipe.named_steps["enc"]
    return {
        "numeric": list(num), "categorical": list(cat),
        "center": scale.center_.tolist(), "scale": scale.scale_.tolist(),
        "categorical_fill": [str(v) for v in imp.statistics_],
        "levels": [[str(v) for v in c] for c in enc.categories_],
        "dropped": [None if d is None else int(d) for d in enc.drop_idx_],
        "features_out": [str(f) for f in ct.get_feature_names_out()],
        "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
    }


def design(m: dict, X) -> np.ndarray:
    """X: DataFrame with the INPUTS columns -> the model's design matrix."""
    X = ClinicalFeatures().transform(X)
    num = X[m["numeric"]].to_numpy(dtype=float)
    if np.isnan(num).any():
        bad = [c for c, b in zip(m["numeric"], np.isnan(num).any(0), strict=True) if b]
        raise ValueError(f"numeric inputs must be complete for the portable model: {bad}")
    cols = [(num - np.array(m["center"])) / np.array(m["scale"])]
    for c, fill, levels, drop in zip(m["categorical"], m["categorical_fill"], m["levels"],
                                     m["dropped"], strict=True):
        v = X[c].astype(object).where(X[c].notna(), fill).astype(str).to_numpy()
        keep = [lv for i, lv in enumerate(levels) if i != drop]
        cols.append(np.column_stack([v == lv for lv in keep]).astype(float))   # unseen -> zeros
    return np.hstack(cols)


def predict_proba(m: dict, X) -> np.ndarray:
    z = design(m, X) @ np.array(m["coef"]) + m["intercept"]
    return 1 / (1 + np.exp(-z))


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
