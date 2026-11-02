"""Post-test exploratory analyses, NOT used for any decision.

Reads only the saved test predictions (results/test_2017_2018/test_predictions.csv). No model is
fitted, refitted or rescored, and results/test_2017_2018/ is never written to.

  1. Primary model vs the ADA-style score: paired bootstrap of the ROC-AUC and PR-AUC
     differences (unweighted and survey-weighted), on the same resamples for both models.
  2. Men vs women, primary model: positives per sex, bootstrap of the women - men difference
     (resampled within each sex), and calibration by sex (slope, intercept, O/E, risk quintiles).

    python scripts/post_test_exploratory.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so `src` imports work

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src.evaluation.metrics import calibration_stats  # noqa: E402

PRED = Path("results/test_2017_2018/test_predictions.csv")
OUT = Path("results/post_test_exploratory")
PRIMARY, ADA = "p_logreg__none__A", "p_ada_style__none__A"
N_BOOT, SEED = 1000, 42
LABEL = "post-test, exploratory, not used for any decision"


def ci(x) -> tuple[float, float]:
    return tuple(np.percentile(x, [2.5, 97.5]))


def disc(y, p, w=None) -> dict:
    return {"roc_auc": roc_auc_score(y, p, sample_weight=w),
            "pr_auc": average_precision_score(y, p, sample_weight=w)}


def calib(y, p) -> dict:
    c = calibration_stats(y, p)
    return {"slope": c["slope"], "intercept": c["intercept"],
            "intercept_offset": c["intercept_offset"], "oe": c["citl_ratio_observed_expected"]}


def primary_vs_ada(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    y, w = df.y.to_numpy(), df.wtmec2yr.to_numpy()
    rows = []
    for weighting, ww in [("unweighted", None), ("WTMEC2YR", w)]:
        a, b = disc(y, df[PRIMARY], ww), disc(y, df[ADA], ww)
        boot = {m: [] for m in a}
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() == y[i].max():
                continue
            wi = None if ww is None else ww[i]
            pa, pb = disc(y[i], df[PRIMARY].to_numpy()[i], wi), disc(y[i], df[ADA].to_numpy()[i], wi)
            for m in boot:
                boot[m].append(pa[m] - pb[m])
        for m in a:
            lo, hi = ci(boot[m])
            rows.append({"comparison": "primary - ADA-style", "weighting": weighting, "metric": m,
                         "primary": a[m], "ada_style": b[m], "difference": a[m] - b[m],
                         "ci_low": lo, "ci_high": hi, "n_boot": len(boot[m]), "label": LABEL})
    return pd.DataFrame(rows)


def by_sex(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    g = {s: (d.y.to_numpy(), d[PRIMARY].to_numpy()) for s, d in df.groupby("sex")}
    point = {s: {**disc(y, p), **calib(y, p)} for s, (y, p) in g.items()}
    boot = {s: {m: [] for m in point[s]} for s in g}
    for _ in range(N_BOOT):
        draw = {}
        for s, (y, p) in g.items():
            i = rng.integers(0, len(y), len(y))
            draw[s] = {**disc(y[i], p[i]), **calib(y[i], p[i])}
        for s in g:
            for m in point[s]:
                boot[s][m].append(draw[s][m])

    groups = []
    for s, (y, p) in g.items():
        row = {"sex": s, "n": len(y), "n_pos": int(y.sum()), "prevalence": y.mean(),
               "mean_predicted": p.mean()}
        for m, v in point[s].items():
            row[m], (row[f"{m}_ci_low"], row[f"{m}_ci_high"]) = v, ci(boot[s][m])
        groups.append(row | {"label": LABEL})

    diffs = []
    for m in point["female"]:
        d = np.subtract(boot["female"][m], boot["male"][m])
        lo, hi = ci(d)
        diffs.append({"metric": m, "female": point["female"][m], "male": point["male"][m],
                      "female_minus_male": point["female"][m] - point["male"][m],
                      "ci_low": lo, "ci_high": hi, "n_boot": N_BOOT, "label": LABEL})

    quint = []
    for s, d in df.groupby("sex"):
        q = pd.qcut(d[PRIMARY], 5, labels=False)
        for k, b in d.groupby(q):
            quint.append({"sex": s, "risk_quintile": int(k) + 1, "n": len(b),
                          "n_pos": int(b.y.sum()), "mean_predicted": b[PRIMARY].mean(),
                          "observed": b.y.mean(), "label": LABEL})
    return pd.DataFrame(groups), pd.DataFrame(diffs), pd.DataFrame(quint)


if __name__ == "__main__":
    df = pd.read_csv(PRED)
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {"primary_vs_ada_paired.csv": primary_vs_ada(df)}
    tables["sex_groups.csv"], tables["sex_difference.csv"], tables["sex_calibration_quintiles.csv"] = by_sex(df)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        for name, tab in tables.items():
            tab.to_csv(OUT / name, index=False, float_format="%.5f")
            print(f"\n{name} ({LABEL})\n{tab.drop(columns='label').round(4).to_string(index=False)}")
