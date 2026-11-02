"""Explanation analysis on raw input features (docs/analysis_plan.md, deviation 2026-11-02 B).

Explanations are of the whole fitted pipeline as a function of its raw inputs. Missing inputs
are filled with the pipeline's own fitted imputers, so every explained value is an imputed input.
Categorical inputs are integer-coded for SHAP/LIME and decoded before prediction.
Counterfactuals explain the model; they are not clinical advice or causal claims.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

NOTE = "model explanation only; not clinical advice or a causal claim"


def imputed_raw(pipe, X: pd.DataFrame) -> pd.DataFrame:
    """Raw input features with missing values filled by the pipeline's fitted imputers."""
    Z = pipe.named_steps["features"].transform(X) if "features" in pipe.named_steps else X
    ct = pipe.named_steps["columns"]
    out = X.copy()
    for name, cols in [("num", ct.transformers_[0][2]), ("cat", ct.transformers_[1][2])]:
        if not len(cols):
            continue
        filled = ct.named_transformers_[name].named_steps["impute"].transform(Z[cols])
        filled = pd.DataFrame(filled, columns=cols, index=X.index)
        for c in X.columns.intersection(cols):
            out[c] = filled[c].astype(object if name == "cat" else float)
    return out


class Codec:
    """Integer codes for categorical inputs, so explainers see a numeric matrix."""

    def __init__(self, train: pd.DataFrame, categorical: list[str]):
        self.columns = list(train.columns)
        self.categorical = [c for c in categorical if c in self.columns]
        self.levels = {c: sorted(train[c].dropna().astype(str).unique()) for c in self.categorical}
        self.cat_idx = [self.columns.index(c) for c in self.categorical]

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        out = df[self.columns].copy()
        for c in self.categorical:
            out[c] = out[c].astype(str).map({v: i for i, v in enumerate(self.levels[c])})
        return out.to_numpy(dtype=float)

    def decode(self, a: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(np.asarray(a, dtype=float), columns=self.columns)
        for c in self.categorical:
            lv = self.levels[c]
            df[c] = [lv[int(np.clip(round(v), 0, len(lv) - 1))] for v in df[c]]
        return df

    def predictor(self, pipe):
        return lambda a: pipe.predict_proba(self.decode(a))[:, 1]


def shap_values(pipe, codec: Codec, X: np.ndarray, background: np.ndarray, seed: int = 42):
    """Permutation SHAP on the probability scale, independent masker over `background`."""
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.explainers.Permutation(codec.predictor(pipe), masker, seed=seed)
    return explainer(X, silent=True).values


def lime_weights(pipe, codec: Codec, train: np.ndarray, X: np.ndarray, seed: int = 42,
                 num_samples: int = 5000) -> np.ndarray:
    """LIME weights (class 1) for every feature, library defaults (quartile discretisation)."""
    f = codec.predictor(pipe)
    explainer = LimeTabularExplainer(train, feature_names=codec.columns, mode="classification",
                                     categorical_features=codec.cat_idx, random_state=seed)
    out = np.zeros_like(X, dtype=float)
    for r, x in enumerate(X):
        exp = explainer.explain_instance(x, lambda a: np.column_stack([1 - f(a), f(a)]),
                                         labels=(1,), num_features=X.shape[1],
                                         num_samples=num_samples)
        for j, w in exp.as_map()[1]:
            out[r, j] = w
    return out


def agreement(shap_v: np.ndarray, lime_w: np.ndarray, k: int = 5) -> pd.DataFrame:
    """Per patient: top-k overlap by |attribution| and Spearman of |SHAP| vs |LIME|."""
    rows = []
    for s, w in zip(np.abs(shap_v), np.abs(lime_w), strict=True):
        top_s, top_w = set(np.argsort(-s)[:k]), set(np.argsort(-w)[:k])
        rho = spearmanr(s, w).statistic if s.std() > 0 and w.std() > 0 else np.nan
        rows.append({"top5_overlap": len(top_s & top_w) / k, "spearman": rho})
    return pd.DataFrame(rows)


def bootstrap_mean(v, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Mean and percentile 95 % CI, resampling the values (patients)."""
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return float(v.mean()), *map(float, np.percentile(means, [2.5, 97.5]))


def pairwise_spearman(importance: np.ndarray) -> np.ndarray:
    """Upper-triangle Spearman correlations between models' importance vectors (rows)."""
    rho = spearmanr(importance.T).statistic
    rho = np.array([[rho]]) if np.ndim(rho) == 0 else rho
    return rho[np.triu_indices(len(importance), k=1)]


def stability(phis: np.ndarray, features: list[str], n_boot: int = 1000, seed: int = 42):
    """phis: (models, patients, features) SHAP values on a common patient set. Returns
    (summary dict, per-feature rank table)."""
    imp = np.abs(phis).mean(axis=1)
    pw = pairwise_spearman(imp)
    rng = np.random.default_rng(seed)
    n = phis.shape[1]
    boot = [pairwise_spearman(np.abs(phis[:, rng.integers(0, n, n)]).mean(axis=1)).mean()
            for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    ranks = (-imp).argsort(axis=1).argsort(axis=1) + 1          # 1 = most important
    table = pd.DataFrame({"feature": features, "mean_abs_shap": imp.mean(axis=0),
                          "rank_median": np.median(ranks, axis=0), "rank_min": ranks.min(axis=0),
                          "rank_max": ranks.max(axis=0)}).sort_values("rank_median")
    return ({"n_models": len(imp), "n_pairs": len(pw), "spearman_mean": float(pw.mean()),
             "spearman_ci_low": float(lo), "spearman_ci_high": float(hi),
             "spearman_min": float(pw.min()), "spearman_max": float(pw.max())}, table)


def cf_metrics(query: pd.Series, cf: pd.DataFrame, p_cf: np.ndarray, train: pd.DataFrame,
               numeric: list[str], history: list[str], threshold: float) -> pd.DataFrame:
    """Validity, proximity (|change| / training MAD, MAD 0 -> 1), sparsity and plausibility
    (changed values inside the training 1st-99th percentile, no history item changed, smoking
    status never moved to 'never')."""
    mad = (train[numeric] - train[numeric].median()).abs().median().replace(0, 1.0)
    q01, q99 = train[numeric].quantile(0.01), train[numeric].quantile(0.99)
    rows = []
    for (_, c), p in zip(cf.iterrows(), p_cf, strict=True):
        changed = [f for f in query.index
                   if (abs(float(c[f]) - float(query[f])) > 1e-6 if f in numeric
                       else str(c[f]) != str(query[f]))]
        num_changed = [f for f in changed if f in numeric]
        in_range = all(q01[f] <= float(c[f]) <= q99[f] for f in num_changed)
        no_history = not any(f in history for f in changed) and not (
            "smoking_status" in changed and str(c["smoking_status"]) == "never")
        rows.append({"valid": bool(p < threshold), "p_cf": float(p),
                     "proximity": (float(np.mean([abs(float(c[f]) - float(query[f])) / mad[f]
                                                  for f in num_changed]))
                                   if num_changed else np.nan),
                     "sparsity": len(changed), "changed": ";".join(changed),
                     "in_range": in_range, "no_history_change": no_history,
                     "plausible": in_range and no_history})
    return pd.DataFrame(rows)
