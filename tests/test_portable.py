"""Portable JSON model: equals the sklearn pipeline on synthetic data; the committed demo runs."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from scripts.demo import SAMPLE, score
from src.models.portable import INPUTS, export, predict_proba
from src.preprocessing.pipeline import build_model_pipeline

CAT = ["sex", "family_history_diabetes", "smoker", "race_eth"]


def synthetic(n=800, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "age": rng.uniform(20, 80, n), "sex": rng.choice(["female", "male"], n),
        "bmi": rng.normal(29, 6, n), "waist_cm": rng.normal(98, 15, n),
        "height_cm": rng.normal(168, 10, n), "systolic_bp": rng.normal(122, 17, n),
        "diastolic_bp": rng.normal(71, 11, n),
        "family_history_diabetes": rng.choice(["no", "yes"], n),
        "smoker": rng.choice(["no", "yes"], n),
        "race_eth": rng.choice(["1", "2", "3", "4", "6", "7"], n)})
    y = (rng.random(n) < 1 / (1 + np.exp(-(-6 + 0.05 * X.age + 0.05 * X.bmi)))).astype(int)
    return X[INPUTS], y


def test_portable_equals_sklearn_on_complete_rows():
    X, y = synthetic()
    pipe = build_model_pipeline(X, LogisticRegression(C=0.3, max_iter=2000), CAT,
                                encoding="onehot").fit(X, y)
    m = export(pipe)
    new = synthetic(200, seed=1)[0]
    new.loc[:5, "race_eth"] = "9"                          # unseen level -> all zeros, as sklearn
    new.loc[6:10, "smoker"] = np.nan                       # categorical fill, as sklearn
    assert np.abs(predict_proba(m, new) - pipe.predict_proba(new)[:, 1]).max() < 1e-12


def test_portable_refuses_missing_numeric():
    X, y = synthetic()
    m = export(build_model_pipeline(X, LogisticRegression(max_iter=2000), CAT,
                                    encoding="onehot").fit(X, y))
    X.loc[0, "bmi"] = np.nan
    with pytest.raises(ValueError, match="bmi"):
        predict_proba(m, X)


def test_demo_scores_sample_patient():
    import json
    p = score(json.loads(SAMPLE.read_text()))
    assert 0 < p < 1
