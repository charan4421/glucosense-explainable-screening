"""Every model must behave as an sklearn classifier, or CV scorers silently fail on it."""
import numpy as np
import pytest
from sklearn.base import is_classifier
from sklearn.metrics import get_scorer

from src.models.models import get_model
from src.preprocessing.pipeline import build_model_pipeline
from tests.test_pipeline import CAT, make_data

FAST = {"mlp": {"model__epochs": 2}, "tabnet": {"model__max_epochs": 2, "model__patience": 1},
        "xgboost": {"model__n_estimators": 20}}


@pytest.mark.parametrize("name", ["xgboost", "mlp", "tabnet"])
def test_model_is_classifier_and_scores(name):
    model, _ = get_model(name, seed=0)
    assert is_classifier(model), f"{name} is not recognised as a classifier by sklearn"
    X, y = make_data(n=400)
    pipe = build_model_pipeline(X, model, CAT, "none").set_params(**FAST[name])
    pipe.fit(X, y)
    score = get_scorer("average_precision")(pipe, X, y)   # failed with (n, 2) y before the fix
    assert np.isfinite(score) and 0 <= score <= 1
