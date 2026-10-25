"""Synthetic-data tests: run without any downloads."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluation.metrics import decision_curve, summarize
from src.evaluation.nested_cv import nested_cv
from src.models.models import make_xgb
from src.preprocessing.pipeline import build_model_pipeline

CAT = ["sex", "smoker"]


def make_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "age": rng.integers(20, 80, n).astype(float),
        "bmi": rng.normal(28, 5, n),
        "waist_cm": rng.normal(95, 12, n),
        "height_cm": rng.normal(170, 9, n),
        "sex": rng.choice(["male", "female"], n),
        "smoker": rng.choice(["yes", "no"], n),
    })
    logit = 0.04 * (X.age - 50) + 0.1 * (X.bmi - 28) - 2.2
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X.loc[rng.random(n) < 0.1, "bmi"] = np.nan          # missing values
    return X, y


def test_pipeline_runs_for_all_strategies():
    X, y = make_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=0)
    for strategy in ["none", "class_weight", "smotenc"]:
        pipe = build_model_pipeline(Xtr, make_xgb(), CAT, strategy)
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        assert p.shape == (len(Xte),) and np.isfinite(p).all()
        assert 0 <= summarize(yte, p)["brier"] <= 1


def test_nested_cv_returns_one_row_per_outer_fold():
    X, y = make_data()
    res, oof = nested_cv(X, y, "xgboost", "none", CAT, outer_folds=3, inner_folds=2, n_iter=2)
    assert len(res) == 3 and len(oof) == len(y)


def test_decision_curve_shapes():
    X, y = make_data()
    dca = decision_curve(y, np.full(len(y), y.mean()))
    assert len(dca["model"]) == len(dca["thresholds"])
