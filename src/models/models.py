"""Model wrappers. All expose fit / predict_proba, so they plug into sklearn CV and search."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class XGB(XGBClassifier):
    """XGBClassifier with balance=True: positives weighted by the neg/pos ratio of the training fold."""

    def __init__(self, balance: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.balance = balance

    def get_xgb_params(self):
        params = super().get_xgb_params()
        params.pop("balance", None)
        return params

    def fit(self, X, y, sample_weight=None, **kw):
        if self.balance:
            y_arr = np.asarray(y)
            pos = max(int(y_arr.sum()), 1)
            w = np.where(y_arr == 1, (len(y_arr) - pos) / pos, 1.0)
            sample_weight = w if sample_weight is None else w * np.asarray(sample_weight)
        return super().fit(X, y, sample_weight=sample_weight, **kw)


def make_xgb(seed: int = 42) -> XGB:
    return XGB(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
               colsample_bytree=0.8, eval_metric="aucpr", random_state=seed, n_jobs=-1)


XGB_SPACE = {
    "model__n_estimators": [150, 300, 500],
    "model__max_depth": [3, 4, 6],
    "model__learning_rate": [0.02, 0.05, 0.1],
    "model__min_child_weight": [1, 5, 10],
}


# Mixin before BaseEstimator: sklearn >= 1.6 reads estimator tags along the MRO, and with
# BaseEstimator first is_classifier() is False, so scorers pass both proba columns and fail.
class TorchMLP(ClassifierMixin, BaseEstimator):
    """Plain PyTorch MLP baseline with sklearn API."""

    def __init__(self, hidden=(128, 64), dropout=0.2, lr=1e-3, epochs=30, batch_size=512,
                 pos_weight=None, seed=42):
        self.hidden, self.dropout, self.lr = hidden, dropout, lr
        self.epochs, self.batch_size, self.pos_weight, self.seed = epochs, batch_size, pos_weight, seed

    def _net(self, d):
        import torch.nn as nn
        layers, prev = [], d
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed)
        X = torch.tensor(np.asarray(X), dtype=torch.float32)
        y_np = np.asarray(y)
        y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
        self.classes_ = np.array([0, 1])
        self.net_ = self._net(X.shape[1])
        pw = self.pos_weight
        if pw == "auto":
            pw = (len(y_np) - y_np.sum()) / max(y_np.sum(), 1)
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pw], dtype=torch.float32) if pw else None)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        n = len(X)
        for _ in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss_fn(self.net_(X[idx]), y[idx]).backward()
                opt.step()
        return self

    def predict_proba(self, X):
        import torch
        self.net_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net_(torch.tensor(np.asarray(X), dtype=torch.float32)))
        p = p.numpy().ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


MLP_SPACE = {
    "model__hidden": [(64,), (128, 64), (256, 128, 64)],
    "model__dropout": [0.1, 0.2, 0.4],
    "model__lr": [1e-3, 3e-4],
    "model__epochs": [20, 40],
}


class TabNetWrapper(ClassifierMixin, BaseEstimator):
    """sklearn-style wrapper around pytorch-tabnet. Holds out 10% of the training fold for early stopping."""

    def __init__(self, n_d=16, n_a=16, n_steps=4, lr=2e-2, max_epochs=60, patience=10,
                 batch_size=1024, cat_idxs=None, cat_dims=None, seed=42):
        self.n_d, self.n_a, self.n_steps, self.lr = n_d, n_a, n_steps, lr
        self.max_epochs, self.patience, self.batch_size = max_epochs, patience, batch_size
        self.cat_idxs, self.cat_dims, self.seed = cat_idxs, cat_dims, seed

    def fit(self, X, y):
        from pytorch_tabnet.tab_model import TabNetClassifier
        from sklearn.model_selection import train_test_split
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.1, stratify=y, random_state=self.seed)
        self.model_ = TabNetClassifier(
            n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps, seed=self.seed, verbose=0,
            optimizer_params={"lr": self.lr},
            cat_idxs=self.cat_idxs or [], cat_dims=self.cat_dims or [], cat_emb_dim=2)
        self.model_.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric=["auc"],
                        max_epochs=self.max_epochs, patience=self.patience,
                        batch_size=self.batch_size)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=np.float32))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


TABNET_SPACE = {
    "model__n_d": [8, 16, 32],
    "model__n_steps": [3, 4, 5],
    "model__lr": [1e-2, 2e-2],
}


LOGREG_SPACE = {"model__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]}

# Models that take the raw feature frame and have nothing to tune (no preprocessing, no search).
RAW_INPUT_MODELS = {"ada_style"}

# Models whose categorical inputs are one-hot encoded. A linear model cannot use an ordinal code
# of a nominal variable (one slope over arbitrary codes); trees and EBM split it. Deviation
# 2026-11-02 in docs/analysis_plan.md.
ONEHOT_MODELS = {"logreg"}


def categorical_encoding(name: str) -> str:
    return "onehot" if name in ONEHOT_MODELS else "ordinal"


# EBM: 8 outer bags (default 14) on 2 parallel jobs to fit this PC (8 GB RAM); the number of pairwise
# interactions is tuned because it dominates fit time (0 interactions ~4 s, 3x default ~1 min).
EBM_SPACE = {
    "model__interactions": [0, 5, 10],
    "model__learning_rate": [0.005, 0.015, 0.05],
    "model__max_leaves": [2, 3],
    "model__min_samples_leaf": [2, 4, 10],
}


def get_model(name: str, seed: int = 42):
    if name == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(outer_bags=8, n_jobs=2, random_state=seed), EBM_SPACE
    if name == "logreg":
        # L2 is the default penalty; C is tuned. Same preprocessing pipeline as the other models.
        return LogisticRegression(max_iter=2000, random_state=seed), LOGREG_SPACE
    if name == "ada_style":
        from src.models.ada_style import ADAStyleScore
        return ADAStyleScore(), {}
    if name == "xgboost":
        return make_xgb(seed), XGB_SPACE
    if name == "mlp":
        return TorchMLP(seed=seed), MLP_SPACE
    if name == "tabnet":
        return TabNetWrapper(seed=seed), TABNET_SPACE
    raise ValueError(name)
