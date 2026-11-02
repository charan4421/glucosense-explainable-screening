"""Decision curve analysis on out-of-fold predictions, 2011-2016 only, over the pre-specified
1-5 % threshold window (screening range for undiagnosed diabetes), with percentile bootstrap
95 % bands (docs/analysis_plan.md, deviation 2026-11-02). All curves share the same resamples.

    python scripts/dev_dca.py                      # primary model + best main model (rules)
    python scripts/dev_dca.py --runs logreg__none__A ebm__none__C
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

from scripts.dev_subgroups import load_dev_oof, primary_run  # noqa: E402
from scripts.run_dev import SUMMARY  # noqa: E402
from src.data.nhanes import MAIN_LABEL  # noqa: E402
from src.evaluation.metrics import bootstrap_net_benefit, net_benefit, treat_all_net_benefit  # noqa: E402
from src.evaluation.selection import primary_set  # noqa: E402

THRESHOLDS = np.round(np.arange(0.01, 0.05 + 1e-9, 0.005), 4)   # pre-specified: 1 % to 5 %


def dca_table(y, preds: dict[str, np.ndarray], thresholds=THRESHOLDS, n_boot: int = 0,
              seed: int = 42) -> pd.DataFrame:
    """Net benefit per threshold for each model, treat-all and treat-none; with n_boot > 0, adds
    <name>_ci_low / _ci_high for each model, treat-all and <model>_minus_treat_all."""
    y = np.asarray(y)
    out = pd.DataFrame({"threshold": thresholds, "treat_all": treat_all_net_benefit(y, thresholds),
                        "treat_none": 0.0})
    for name, p in preds.items():
        out[name] = net_benefit(y, p, thresholds)
        out[f"{name}_minus_treat_all"] = out[name] - out.treat_all
    if n_boot:
        for name, (lo, hi) in bootstrap_net_benefit(y, preds, thresholds, n_boot, seed).items():
            out[f"{name}_ci_low"], out[f"{name}_ci_high"] = lo, hi
    return out


def default_runs(cfg: dict) -> list[str]:
    best = primary_set(pd.read_csv(SUMMARY), cfg["design"]["main_models"])
    runs = [primary_run(), f"{best['best_model']}__none__{best['best_set']}"]
    return list(dict.fromkeys(runs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--runs", nargs="+", default=None, help="<model>__<strategy>__<set> ...")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    runs = a.runs or default_runs(cfg)
    preds = {}
    for run in runs:
        dev, preds[run] = load_dev_oof(cfg, run)
    y = dev[MAIN_LABEL].to_numpy()
    tab = dca_table(y, preds, n_boot=a.n_boot, seed=cfg["seed"])
    tab.to_csv("results/dev_cv_v2_dca.csv", index=False, float_format="%.5f")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = tab.threshold * 100
    for run in runs:
        line, = ax.plot(x, tab[run], marker="o", label=run.replace("__", " / "))
        ax.fill_between(x, tab[f"{run}_ci_low"], tab[f"{run}_ci_high"], color=line.get_color(),
                        alpha=0.15, lw=0)
    ax.plot(x, tab.treat_all, "k:", label="treat all")
    ax.fill_between(x, tab.treat_all_ci_low, tab.treat_all_ci_high, color="k", alpha=0.08, lw=0)
    ax.plot(x, tab.treat_none, "k-", lw=1, label="treat none")
    top = max(float(tab[f"{r}_ci_high"].max()) for r in runs)
    ax.set(xlabel="Threshold probability (%)", ylabel="Net benefit",
           title=f"Decision curve, out-of-fold 2011-2016 (n={len(y)}, prevalence {y.mean():.1%})\n"
                 f"shaded: bootstrap 95 % bands ({a.n_boot} resamples)",
           ylim=(min(-0.005, float(tab.treat_all_ci_low.min())), top * 1.1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/dev_cv_v2_dca.png", dpi=150)
    print(f"out-of-fold 2011-2016, n={len(y)}, prevalence {y.mean():.4f}, {a.n_boot} resamples "
          "-> results/dev_cv_v2_dca.csv/.png")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(tab.round(5).to_string(index=False))
