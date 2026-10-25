"""Metrics: PR-AUC, ROC-AUC, Brier, calibration, and Decision Curve Analysis."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

METRICS = {"pr_auc": average_precision_score, "roc_auc": roc_auc_score, "brier": brier_score_loss}


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


def _logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def calibration_stats(y_true, p) -> dict:
    """Calibration of predicted risks.

    - citl_*: calibration-in-the-large, mean predicted risk vs observed prevalence.
    - slope / intercept: logistic regression of the outcome on logit(p) (unpenalised).
      Ideal: slope 1, intercept 0. Slope < 1 = predictions too extreme (overfit).
    - intercept_offset: intercept with the slope fixed at 1 (logit(p) as offset), the usual
      "calibration intercept"; > 0 means risks are underestimated on average.
    """
    y, lp = np.asarray(y_true, dtype=float), _logit(p)
    lr = LogisticRegression(C=np.inf, max_iter=1000).fit(lp.reshape(-1, 1), y)   # unpenalised
    # solve sum(y - sigmoid(a + lp)) = 0 for a (intercept-only model with offset lp)
    a = brentq(lambda a: np.sum(y - 1 / (1 + np.exp(-(a + lp)))), -20, 20)
    mean_p, obs = float(np.mean(p)), float(np.mean(y))
    return {"citl_mean_predicted": mean_p, "citl_observed": obs,
            "citl_difference": obs - mean_p, "citl_ratio_observed_expected": obs / mean_p,
            "slope": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0]),
            "intercept_offset": float(a)}


def bootstrap_ci(y_true, p, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CIs (resampling test rows with replacement) for PR-AUC, ROC-AUC, Brier."""
    y, p = np.asarray(y_true), np.asarray(p)
    rng = np.random.default_rng(seed)
    stats = {k: [] for k in METRICS}
    n_skipped = 0
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():          # a resample with one class has no AUC
            n_skipped += 1
            continue
        for k, f in METRICS.items():
            stats[k].append(f(y[idx], p[idx]))
    out = {"n_boot": n_boot, "n_skipped": n_skipped}
    for k, v in stats.items():
        lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[f"{k}_ci_low"], out[f"{k}_ci_high"] = float(lo), float(hi)
    return out


def plot_calibration(y_true, p, path, n_bins: int = 10, title: str = "Calibration") -> None:
    """Quantile-binned reliability plot with Wilson 95% intervals per bin."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y, p = np.asarray(y_true), np.asarray(p)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    mp, ob, lo, hi = [], [], [], []
    for k in range(len(edges) - 1):
        m = b == k
        n, x = m.sum(), y[m].sum()
        f, z = x / n, 1.96
        c = (f + z * z / (2 * n)) / (1 + z * z / n)
        h = z * np.sqrt(f * (1 - f) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        mp.append(p[m].mean())
        ob.append(f)
        lo.append(c - h)
        hi.append(c + h)
    mp, ob = np.array(mp), np.array(ob)
    top = max(mp.max(), np.max(hi)) * 1.1
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, top], [0, top], "k--", lw=1, label="perfect calibration")
    ax.errorbar(mp, ob, yerr=[ob - np.array(lo), np.array(hi) - ob], fmt="o-", capsize=3,
                label=f"{len(mp)} quantile bins (95% CI)")
    ax.set(xlim=(0, top), ylim=(0, top), xlabel="Mean predicted risk", ylabel="Observed fraction",
           title=title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
