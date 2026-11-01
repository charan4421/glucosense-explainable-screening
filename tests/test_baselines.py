"""Baselines: L2 logistic regression and the ADA-style score."""
import numpy as np
import pandas as pd
from sklearn.base import is_classifier

from src.evaluation.nested_cv import nested_cv
from src.models.ada_style import ADAStyleScore, ada_style_points
from src.models.models import get_model
from tests.test_pipeline import CAT, make_data


def ada_frame():
    return pd.DataFrame({
        "age": [35, 45, 55, 65, 70, 30],
        "sex": ["female", "male", "female", "male", "male", "female"],
        "family_history_diabetes": ["no", "yes", "no", "yes", np.nan, "no"],
        "systolic_bp": [120, 145, 120, 150, np.nan, 110],
        "diastolic_bp": [70, 80, 95, 85, np.nan, 70],
        "bmi": [22, 27, 33, 42, np.nan, 24.9],
    })


def test_ada_style_points():
    # age + sex + family history + BP proxy + BMI; missing values score 0
    assert ada_style_points(ada_frame()).tolist() == [0, 1 + 1 + 1 + 1 + 1, 2 + 0 + 0 + 1 + 2,
                                                      3 + 1 + 1 + 1 + 3, 3 + 1, 0]


def test_ada_style_risk_is_monotone_in_points():
    X = pd.concat([ada_frame()] * 20, ignore_index=True)
    pts = ada_style_points(X)
    y = (pts + np.random.default_rng(0).normal(0, 2, len(X)) > 4).astype(int)
    m = ADAStyleScore().fit(X, y)
    assert is_classifier(m)
    p = m.predict_proba(X)[:, 1]
    order = np.argsort(pts)
    assert np.all(np.diff(p[order]) >= -1e-12)


def test_logreg_is_l2_and_runs_in_nested_cv():
    model, space = get_model("logreg")
    assert is_classifier(model) and model.get_params()["C"] == 1.0 and "model__C" in space
    X, y = make_data(n=400)
    res, oof = nested_cv(X, y, "logreg", "none", CAT, outer_folds=3, inner_folds=2, n_iter=2)
    assert len(res) == 3 and res.pr_auc.between(0, 1).all() and ((oof > 0) & (oof < 1)).all()
