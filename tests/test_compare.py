"""Nadeau-Bengio corrected paired comparison on identical folds."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.evaluation.compare import comparison_table, nadeau_bengio, paired


def test_nadeau_bengio_matches_formula_and_is_wider_than_naive():
    d = np.array([0.01, 0.02, -0.005, 0.015, 0.0, 0.012, 0.008, -0.002, 0.02, 0.01,
                  0.005, 0.0, 0.018, 0.003, 0.009])              # 5 folds x 3 repeats
    r = nadeau_bengio(d, 1 / 4)
    se = np.sqrt((1 / 15 + 0.25) * d.var(ddof=1))
    half = stats.t.ppf(0.975, 14) * se
    assert np.isclose(r["ci_low"], d.mean() - half) and np.isclose(r["ci_high"], d.mean() + half)
    naive = stats.t.ppf(0.975, 14) * d.std(ddof=1) / np.sqrt(15)
    assert half > naive


def folds(model, set_, scores, n=None):
    return pd.DataFrame({"model": model, "feature_set": set_, "strategy": "none",
                         "repeat": np.repeat([0, 1, 2], 5), "fold": np.tile(range(5), 3),
                         "pr_auc": scores, "roc_auc": 0.8, "brier": 0.03,
                         "n": n if n is not None else 2600})


def test_pairing_requires_identical_folds():
    rng = np.random.default_rng(0)
    a, b = folds("xgboost", "A", rng.normal(0.1, 0.01, 15)), folds("logreg", "A", 0.09)
    r = paired(pd.concat([a, b]), {"model": "xgboost"}, {"model": "logreg"}, "pr_auc", 5)
    assert r["n_folds"] == 15
    bad = folds("logreg", "A", 0.09, n=np.r_[[2500], [2600] * 14])
    with pytest.raises(ValueError, match="identical folds"):
        paired(pd.concat([a, bad]), {"model": "xgboost"}, {"model": "logreg"}, "pr_auc", 5)


def test_comparison_table_covers_model_and_set_pairs():
    f = pd.concat([folds(m, s, 0.1) for m in ["logreg", "xgboost"] for s in "AB"])
    t = comparison_table(f, ["logreg", "xgboost"], ["A", "B"], 5, metrics=("pr_auc",))
    assert (t.comparison == "model").sum() == 2 and (t.comparison == "set").sum() == 2
