"""One-hot encoding for logistic regression (deviation 2026-11-02): pipeline output and hash."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluation.checkpoint import combo_hash
from src.models.models import categorical_encoding
from src.preprocessing.pipeline import build_model_pipeline, build_preprocessor


def _X(n=60):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"age": rng.uniform(20, 80, n),
                         "sex": rng.choice(["female", "male"], n),
                         "race_eth": rng.choice(["1", "2", "3", "4", "6", "7"], n)})


def test_only_logreg_is_onehot():
    assert categorical_encoding("logreg") == "onehot"
    assert {categorical_encoding(m) for m in ["xgboost", "ebm", "mlp", "tabnet"]} == {"ordinal"}


def test_onehot_gives_one_column_per_level_and_binary_stays_single():
    X = _X()
    ordinal = build_preprocessor(X, ["sex", "race_eth"], interactions=False).fit_transform(X)
    onehot = build_preprocessor(X, ["sex", "race_eth"], interactions=False,
                                encoding="onehot").fit_transform(X)
    assert ordinal.shape[1] == 3                      # age, sex code, race code
    assert onehot.shape[1] == 1 + 1 + 6               # age, sex 0/1, six race dummies
    assert np.allclose(onehot[:, 2:].sum(axis=1), 1)  # exactly one race level per row
    assert np.array_equal(onehot[:, 1], ordinal[:, 1])   # binary item unchanged in effect


def test_unseen_level_and_smote_guard():
    X = _X()
    pipe = build_model_pipeline(X, LogisticRegression(), ["sex", "race_eth"], interactions=False,
                                encoding="onehot").fit(X, (X.age > 50).astype(int))
    new = X.head(1).assign(race_eth="9")              # unseen level -> all race dummies zero
    assert 0 < pipe.predict_proba(new)[0, 1] < 1
    with pytest.raises(ValueError):
        build_model_pipeline(X, LogisticRegression(), ["sex"], "smotenc", encoding="onehot")


def test_hash_unchanged_for_ordinal_and_changed_for_onehot(tmp_path):
    cfg = {"cv": {"outer_folds": 5}, "preprocessing": {"knn_neighbors": 5},
           "features": {"interactions": True}, "seed": 42}
    data = tmp_path / "d.parquet"
    data.write_bytes(b"x")
    base = combo_hash(cfg, ["age"], [], 5, data)
    assert combo_hash(cfg, ["age"], [], 5, data, "ordinal") == base
    assert combo_hash(cfg, ["age"], [], 5, data, "onehot") != base
