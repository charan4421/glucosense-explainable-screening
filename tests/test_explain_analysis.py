"""Explanation analysis helpers: imputation, coding, agreement, stability, counterfactual metrics."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.explain.analysis import (
    Codec,
    agreement,
    cf_metrics,
    imputed_raw,
    shap_values,
    stability,
)
from src.preprocessing.pipeline import build_model_pipeline


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"age": rng.uniform(20, 80, n), "bmi": rng.normal(28, 5, n),
                      "sex": rng.choice(["male", "female"], n)})
    y = (rng.random(n) < 1 / (1 + np.exp(-(X.age - 50) / 10))).astype(int)
    X.loc[:9, "bmi"] = np.nan
    X.loc[10:14, "sex"] = np.nan
    return X, y


def _pipe(X, y):
    return build_model_pipeline(X, LogisticRegression(), ["sex"], interactions=False).fit(X, y)


def test_imputed_raw_fills_with_fitted_imputers():
    X, y = _data()
    Z = imputed_raw(_pipe(X, y), X)
    assert Z.notna().all().all() and list(Z.columns) == list(X.columns)
    assert (Z.age == X.age).all() and Z.sex.isin(["male", "female"]).all()


def test_codec_roundtrip_and_shap_sums_to_prediction_gap():
    X, y = _data()
    pipe = _pipe(X, y)
    Z = imputed_raw(pipe, X)
    codec = Codec(Z, ["sex"])
    A = codec.encode(Z)
    pd.testing.assert_frame_equal(codec.decode(A), Z.reset_index(drop=True), check_dtype=False)
    phi = shap_values(pipe, codec, A[:5], A[100:150])
    f = codec.predictor(pipe)
    assert np.allclose(phi.sum(1), f(A[:5]) - f(A[100:150]).mean(), atol=1e-6)   # efficiency
    assert np.abs(phi[:, 0]).mean() > np.abs(phi[:, 1]).mean()                  # age drives y


def test_agreement_metrics():
    s = np.array([[5, 4, 3, 2, 1, 0.5], [1, 2, 3, 4, 5, 6.0]])
    t = agreement(s, s * -1)                         # same magnitudes, any sign
    assert (t.top5_overlap == 1).all() and np.allclose(t.spearman, 1)
    t = agreement(s[:1], s[1:])                      # reversed ranking
    assert t.top5_overlap[0] == 0.8 and np.isclose(t.spearman[0], -1)


def test_stability_identical_models_correlate_perfectly():
    phi = np.random.default_rng(0).normal(size=(1, 30, 4)) * np.array([4, 3, 2, 1])
    summary, table = stability(np.repeat(phi, 3, axis=0), list("abcd"), n_boot=20)
    assert summary["n_pairs"] == 3 and np.isclose(summary["spearman_mean"], 1)
    assert list(table.feature) == list("abcd") and (table.rank_min == table.rank_max).all()


def test_cf_metrics():
    train = pd.DataFrame({"bmi": np.linspace(20, 40, 101), "htn_told": [0.0, 1.0] * 50 + [0.0],
                          "smoking_status": ["current"] * 101})
    q = pd.Series({"bmi": 35.0, "htn_told": 1.0, "smoking_status": "current"})
    cf = pd.DataFrame([{"bmi": 30.0, "htn_told": 1.0, "smoking_status": "current"},
                       {"bmi": 30.0, "htn_told": 0.0, "smoking_status": "never"},
                       {"bmi": 45.0, "htn_told": 1.0, "smoking_status": "former"}])
    m = cf_metrics(q, cf, np.array([0.02, 0.01, 0.05]), train, ["bmi", "htn_told"], ["htn_told"],
                   0.03)
    assert list(m.valid) == [True, True, False] and list(m.sparsity) == [1, 3, 2]
    assert list(m.plausible) == [True, False, False]
    assert not m.in_range[2] and not m.no_history_change[1]
    assert np.isclose(m.proximity[0], 5 / 5.0)       # |35-30| / MAD(bmi) = 5 / 5
