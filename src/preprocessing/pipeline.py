"""Leakage-safe preprocessing pipeline.

Everything (imputation, scaling, SMOTE-NC) lives inside one imblearn Pipeline, so cross-validation
fits it on the training fold only. SMOTE is skipped automatically at predict time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler


class ClinicalFeatures(BaseEstimator, TransformerMixin):
    """Adds clinical interaction terms. Stateless, so it cannot leak."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if {"age", "bmi"} <= set(X.columns):
            X["age_x_bmi"] = X["age"] * X["bmi"]
            X["bmi_per_age"] = X["bmi"] / X["age"].clip(lower=1)
        if {"waist_cm", "height_cm"} <= set(X.columns):
            X["waist_height_ratio"] = X["waist_cm"] / X["height_cm"]
        if {"systolic_bp", "diastolic_bp"} <= set(X.columns):
            X["pulse_pressure"] = X["systolic_bp"] - X["diastolic_bp"]
        return X


def _split_columns(X: pd.DataFrame, categorical: list[str]) -> tuple[list[str], list[str]]:
    cat = [c for c in categorical if c in X.columns]
    num = [c for c in X.columns if c not in cat]
    return num, cat


def _encoder(encoding: str):
    if encoding == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if encoding == "onehot":     # binary items stay one 0/1 column; unseen levels -> all zeros
        return OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)
    raise ValueError(f"unknown categorical encoding {encoding!r}")


def build_preprocessor(X: pd.DataFrame, categorical: list[str], knn_neighbors: int = 5,
                       interactions: bool = True, encoding: str = "ordinal") -> Pipeline:
    """Returns a transformer-only pipeline. Output column order: numeric first, then categorical
    (one column per categorical input when ordinal, one per level when one-hot)."""
    X0 = ClinicalFeatures().fit_transform(X) if interactions else X
    num, cat = _split_columns(X0, categorical)
    ct = ColumnTransformer(
        [
            ("num", Pipeline([
                ("impute", KNNImputer(n_neighbors=knn_neighbors)),
                ("scale", RobustScaler()),
            ]), num),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("enc", _encoder(encoding)),
            ]), cat),
        ],
        verbose_feature_names_out=False,
    )
    steps = []
    if interactions:
        steps.append(("features", ClinicalFeatures()))
    steps.append(("columns", ct))
    pipe = Pipeline(steps)
    pipe.feature_names_ = num + cat
    pipe.n_numeric_ = len(num)
    pipe.n_categorical_ = len(cat)
    pipe.encoding_ = encoding
    return pipe


def build_model_pipeline(X: pd.DataFrame, model, categorical: list[str], strategy: str = "none",
                         knn_neighbors: int = 5, interactions: bool = True,
                         seed: int = 42, encoding: str = "ordinal") -> Pipeline:
    """Preprocessing -> optional SMOTE-NC -> model, as one estimator safe for CV."""
    pre = build_preprocessor(X, categorical, knn_neighbors, interactions, encoding)
    steps = list(pre.steps)
    model = clone(model)
    if strategy == "smotenc" and encoding != "ordinal":
        raise ValueError("SMOTE-NC needs one column per categorical input (ordinal encoding)")
    if strategy == "smotenc" and pre.n_categorical_ > 0:
        cat_idx = list(range(pre.n_numeric_, pre.n_numeric_ + pre.n_categorical_))
        steps.append(("smote", SMOTENC(categorical_features=cat_idx, random_state=seed)))
    elif strategy == "class_weight":
        _apply_class_weight(model)
    steps.append(("model", model))
    pipe = Pipeline(steps)
    pipe.feature_names_ = pre.feature_names_
    return pipe


def _apply_class_weight(model) -> None:
    """Sets the imbalance weight on models that support it. Weight is set in fit via pos_weight."""
    if hasattr(model, "set_params"):
        params = model.get_params()
        if "balance" in params:
            model.set_params(balance=True)                  # handled inside our XGB wrapper
        elif "class_weight" in params:
            model.set_params(class_weight="balanced")
        elif "pos_weight" in params:
            model.set_params(pos_weight="auto")


def to_numpy(X) -> np.ndarray:
    return X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
