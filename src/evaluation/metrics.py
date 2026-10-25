"""Metrics: PR-AUC, ROC-AUC, Brier, calibration, and Decision Curve Analysis."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def summarize(y_true, p) -> dict:
    y_true = np.asarray(y_true)
    return {
        "pr_auc": average_precision_score(y_true, p),
        "roc_auc": roc_auc_score(y_true, p),
        "brier": brier_score_loss(y_true, p),
        "prevalence": float(y_true.mean()),
    }


def calibration_table(y_true, p, n_bins: int = 10):
    frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="quantile")
    return mean_pred, frac_pos


def net_benefit(y_true, p, thresholds: np.ndarray) -> np.ndarray:
    """Net benefit of 'treat if p >= t': TP/n - FP/n * t/(1-t)."""
    y_true, p = np.asarray(y_true), np.asarray(p)
    n = len(y_true)
    out = []
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        out.append(tp / n - fp / n * (t / (1 - t)))
    return np.array(out)


def decision_curve(y_true, p, thresholds=None) -> dict:
    thresholds = np.linspace(0.01, 0.5, 50) if thresholds is None else thresholds
    prev = float(np.mean(y_true))
    return {
        "thresholds": thresholds,
        "model": net_benefit(y_true, p, thresholds),
        "treat_all": prev - (1 - prev) * thresholds / (1 - thresholds),
        "treat_none": np.zeros_like(thresholds),
    }
