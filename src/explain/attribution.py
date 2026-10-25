"""SHAP (global + local) and LIME consistency check. Fits on the preprocessed matrix of a tree model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr


def transform(pipe, X: pd.DataFrame) -> pd.DataFrame:
    """Applies every pipeline step except SMOTE and the model."""
    Z = X
    for name, step in pipe.steps:
        if name in ("smote", "model"):
            break
        Z = step.transform(Z)
    return pd.DataFrame(Z, columns=pipe.feature_names_, index=X.index)


def shap_global(pipe, X: pd.DataFrame, n: int = 2000, seed: int = 42):
    """TreeExplainer on the XGBoost model. Returns (explainer, shap_values, Z sample)."""
    Z = transform(pipe, X).sample(min(n, len(X)), random_state=seed)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    return explainer, explainer.shap_values(Z), Z


def shap_dependence(shap_values, Z: pd.DataFrame, feature: str, interaction: str = "auto", ax=None):
    shap.dependence_plot(feature, shap_values, Z, interaction_index=interaction, ax=ax, show=False)


def shap_waterfall(explainer, shap_values, Z: pd.DataFrame, row: int):
    exp = shap.Explanation(values=shap_values[row], base_values=explainer.expected_value,
                           data=Z.iloc[row].values, feature_names=list(Z.columns))
    shap.plots.waterfall(exp, show=False)


def lime_vs_shap(pipe, X: pd.DataFrame, rows: list[int], shap_values, Z: pd.DataFrame,
                 categorical_idx: list[int] | None = None, k: int = 8, seed: int = 42):
    """For each row, compare LIME and SHAP feature rankings (Spearman on |attribution|)."""
    model = pipe.named_steps["model"]
    explainer = LimeTabularExplainer(
        Z.values, feature_names=list(Z.columns), categorical_features=categorical_idx or [],
        class_names=["no", "yes"], mode="classification", random_state=seed)
    out = []
    for r in rows:
        exp = explainer.explain_instance(Z.iloc[r].values, model.predict_proba,
                                         num_features=len(Z.columns))
        lime_w = dict(exp.as_map()[1])
        lime_abs = np.array([abs(lime_w.get(i, 0.0)) for i in range(len(Z.columns))])
        rho = spearmanr(lime_abs, np.abs(shap_values[r])).correlation
        out.append({"row": r, "spearman_lime_vs_shap": rho})
    return pd.DataFrame(out)
