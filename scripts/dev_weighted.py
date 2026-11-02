"""Survey-weighted discrimination of the primary model, 2011-2016 only: a sensitivity analysis
(docs/analysis_plan.md, deviation 2026-11-02). The unweighted results stay primary.

Scores the existing out-of-fold predictions (no refit) unweighted and weighted with WTMEC2YR
(divided by the number of pooled cycles; undivided within a cycle). Percentile bootstrap 95 %
intervals resample persons, who keep their weight; the NHANES strata and PSUs are not in the
development file, so the intervals ignore the design and are too narrow.

    python scripts/dev_weighted.py                 # primary model on the primary set
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so `src` imports work

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from scripts.dev_subgroups import load_dev_oof, primary_run  # noqa: E402
from src.data.nhanes import MAIN_LABEL  # noqa: E402
from src.evaluation.metrics import bootstrap_ci, summarize  # noqa: E402

OUT = Path("results/dev_cv_v2_weighted.csv")


def weighted_table(dev: pd.DataFrame, oof: np.ndarray, n_boot: int, seed: int) -> pd.DataFrame:
    n_cycles = dev.cycle.nunique()
    scopes = [("pooled", np.ones(len(dev), dtype=bool), n_cycles)]
    scopes += [(c, (dev.cycle == c).to_numpy(), 1) for c in sorted(dev.cycle.unique())]
    rows = []
    for scope, m, divisor in scopes:
        y, p = dev.loc[m, MAIN_LABEL].to_numpy(), oof[m]
        for weighting, w in [("unweighted", None), ("WTMEC2YR", dev.loc[m, "wtmec2yr"].to_numpy()
                                                    / divisor)]:
            s = summarize(y, p, sample_weight=w)
            ci = bootstrap_ci(y, p, n_boot, seed, sample_weight=w)
            rows.append({"scope": scope, "weighting": weighting, "n": int(m.sum()),
                         "n_pos": int(y.sum()), "prevalence": s["prevalence"],
                         "roc_auc": s["roc_auc"], "roc_auc_ci_low": ci["roc_auc_ci_low"],
                         "roc_auc_ci_high": ci["roc_auc_ci_high"],
                         "pr_auc": s["pr_auc"], "pr_auc_ci_low": ci["pr_auc_ci_low"],
                         "pr_auc_ci_high": ci["pr_auc_ci_high"],
                         "pr_auc_over_prevalence": s["pr_auc"] / s["prevalence"],
                         "n_boot": n_boot, "n_boot_skipped": ci["n_skipped"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--run", default=None, help="<model>__<strategy>__<set>; default: primary")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    run = a.run or primary_run()
    dev, oof = load_dev_oof(cfg, run)
    tab = weighted_table(dev, oof, a.n_boot, cfg["seed"]).assign(run=run)
    tab.to_csv(OUT, index=False, float_format="%.5f")
    print(f"{run}, out-of-fold (mean of 3 repeats), 2011-2016 (n={len(dev)}), "
          f"{a.n_boot} resamples, design ignored in the intervals -> {OUT}")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(tab.drop(columns="run").round(4).to_string(index=False))
