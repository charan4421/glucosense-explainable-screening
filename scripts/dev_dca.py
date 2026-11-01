"""Decision curve analysis on out-of-fold predictions, 2011-2016 only, over the pre-specified
1-5 % threshold window (screening range for undiagnosed diabetes).

    python scripts/dev_dca.py --runs xgboost__none logreg__none
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so `src` imports work

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from scripts.dev_subgroups import load_dev_oof  # noqa: E402
from src.data.nhanes import MAIN_LABEL  # noqa: E402
from src.evaluation.metrics import net_benefit  # noqa: E402

THRESHOLDS = np.round(np.arange(0.01, 0.05 + 1e-9, 0.005), 4)   # pre-specified: 1 % to 5 %


def dca_table(y, preds: dict[str, np.ndarray], thresholds=THRESHOLDS) -> pd.DataFrame:
    """Net benefit per threshold for each model, treat-all and treat-none."""
    y = np.asarray(y)
    prev = y.mean()
    out = pd.DataFrame({"threshold": thresholds,
                        "treat_all": prev - (1 - prev) * thresholds / (1 - thresholds),
                        "treat_none": 0.0})
    for name, p in preds.items():
        out[name] = net_benefit(y, p, thresholds)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--runs", nargs="+", default=["xgboost__none", "logreg__none"])
    ap.add_argument("--n-iter", type=int, default=5)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    preds = {}
    for run in a.runs:
        dev, preds[run] = load_dev_oof(cfg, *run.split("__"), a.n_iter)
    y = dev[MAIN_LABEL].to_numpy()
    tab = dca_table(y, preds)
    tab.to_csv("results/dca_dev_oof.csv", index=False, float_format="%.5f")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for run in a.runs:
        ax.plot(tab.threshold * 100, tab[run], marker="o", label=run.replace("__", " / "))
    ax.plot(tab.threshold * 100, tab.treat_all, "k:", label="treat all")
    ax.plot(tab.threshold * 100, tab.treat_none, "k-", lw=1, label="treat none")
    ax.set(xlabel="Threshold probability (%)", ylabel="Net benefit",
           title=f"Decision curve, out-of-fold 2011-2016 (n={len(y)}, prevalence {y.mean():.1%})",
           ylim=(min(-0.005, float(tab.treat_all.min())), float(tab.drop(columns="threshold")
                                                              .to_numpy().max()) * 1.15))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/dca_dev_oof.png", dpi=150)
    print(f"out-of-fold 2011-2016, n={len(y)}, prevalence {y.mean():.4f} -> results/dca_dev_oof.csv/.png")
    print(tab.round(5).to_string(index=False))
