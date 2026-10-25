"""Run the comparison: models x imbalance strategies under nested CV, or the temporal test
(tune on earlier NHANES cycles, score once on data.temporal_test_cycle).

    python scripts/run_experiment.py                                   # nested CV, all from config
    python scripts/run_experiment.py --mode temporal --models xgboost --strategies none
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so `src` imports work

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.data.nhanes import MAIN_LABEL  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    bootstrap_ci,
    calibration_stats,
    calibration_table,
    decision_curve,
    plot_calibration,
)
from src.evaluation.nested_cv import nested_cv  # noqa: E402
from src.evaluation.temporal import feature_columns, temporal_evaluate, temporal_split  # noqa: E402

TARGET = MAIN_LABEL


def run_temporal(cfg: dict, models: list[str], strategies: list[str],
                 n_boot: int = 1000) -> pd.DataFrame:
    """Temporal test per model/strategy: metrics with bootstrap CIs, calibration stats and plot.

    Writes reports/temporal_test.csv (one row per run), temporal_predictions_<run>.csv and
    temporal_calibration_<run>.png."""
    out = Path("reports")
    out.mkdir(exist_ok=True)
    df = pd.read_parquet(cfg["data"]["processed_path"])
    feats, cat = feature_columns(cfg), cfg["features"]["categorical"]
    test_cycle = cfg["data"]["temporal_test_cycle"]
    rows = []
    for model in models:
        for strategy in strategies:
            run = f"{model}_{strategy}"
            print(f"== temporal {model} / {strategy}: test {test_cycle}")
            row, preds = temporal_evaluate(df, feats, cat, TARGET, test_cycle, model, strategy, cfg)
            y, p = preds.y_true.to_numpy(), preds.p.to_numpy()
            row.update(bootstrap_ci(y, p, n_boot=n_boot, seed=cfg["seed"]))
            row.update(calibration_stats(y, p))
            preds.to_csv(out / f"temporal_predictions_{run}.csv", index_label="row")
            plot_calibration(y, p, out / f"temporal_calibration_{run}.png", n_bins=10,
                             title=f"{model}/{strategy}: test {test_cycle} (n={len(y)})")
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(out / "temporal_test.csv", index=False)
    for r in rows:
        print(f"\n{r['model']}/{r['strategy']}  test {r['test_cycle']}  n={r['n_test']}  "
              f"(bootstrap {r['n_boot']} resamples, 95% CI)")
        # PR-AUC is only meaningful against the base rate it has to beat
        print(f"  PR-AUC  {r['pr_auc']:.4f} [{r['pr_auc_ci_low']:.4f}, {r['pr_auc_ci_high']:.4f}]"
              f"   test prevalence {r['test_prevalence']:.4f} "
              f"({r['pr_auc'] / r['test_prevalence']:.1f}x base rate)")
        print(f"  ROC-AUC {r['roc_auc']:.4f} [{r['roc_auc_ci_low']:.4f}, {r['roc_auc_ci_high']:.4f}]")
        print(f"  Brier   {r['brier']:.4f} [{r['brier_ci_low']:.4f}, {r['brier_ci_high']:.4f}]"
              f"   (always-predict-prevalence: "
              f"{r['test_prevalence'] * (1 - r['test_prevalence']):.4f})")
        print(f"  calibration-in-the-large: mean predicted {r['citl_mean_predicted']:.4f} vs "
              f"observed {r['citl_observed']:.4f} (O/E {r['citl_ratio_observed_expected']:.2f})")
        print(f"  calibration slope {r['slope']:.3f}, intercept {r['intercept']:.3f} "
              f"(logistic regression on logit p); intercept with slope fixed at 1: "
              f"{r['intercept_offset']:.3f}")
    return res


SUMMARY_METRICS = ["pr_auc", "roc_auc", "brier", "citl_difference", "citl_ratio_observed_expected",
                   "calibration_slope", "prevalence", "fit_seconds"]


def main(cfg_path: str, models: list[str] | None = None, strategies: list[str] | None = None,
         n_iter: int | None = None) -> None:
    """Nested CV on the development cycles only: the temporal test cycle is removed first."""
    cfg = yaml.safe_load(open(cfg_path))
    out, res_dir = Path("reports"), Path("results")
    out.mkdir(exist_ok=True)
    res_dir.mkdir(exist_ok=True)

    test_cycle = cfg["data"]["temporal_test_cycle"]
    dev, _ = temporal_split(pd.read_parquet(cfg["data"]["processed_path"]), test_cycle)
    assert test_cycle not in set(dev.cycle), "test cycle leaked into nested CV"
    print(f"nested CV on {sorted(dev.cycle.unique())} (n={len(dev)}); {test_cycle} held out")
    feats = feature_columns(cfg)
    X, y = dev[feats].reset_index(drop=True), dev[TARGET].reset_index(drop=True)
    cat = cfg["features"]["categorical"]
    cv = cfg["cv"]
    n_iter = n_iter or cv["n_iter"]

    all_rows, oof_store = [], {}
    for model in models or cfg["models"]:
        for strategy in strategies or cfg["imbalance"]["strategies"]:
            if model == "tabnet" and strategy == "class_weight":
                continue
            print(f"== {model} / {strategy}  (outer {cv['outer_folds']}, inner {cv['inner_folds']}, "
                  f"n_iter {n_iter})", flush=True)
            res, oof = nested_cv(
                X, y, model, strategy, cat, cv["outer_folds"], cv["inner_folds"], n_iter,
                cv["scoring"], cfg["preprocessing"]["knn_neighbors"],
                cfg["features"]["interactions"], cfg["seed"])
            print(f"   {res.fit_seconds.sum():.0f}s  PR-AUC {res.pr_auc.mean():.4f}", flush=True)
            all_rows.append(res)
            oof_store[(model, strategy)] = oof

    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(out / "nested_cv_folds.csv", index=False)
    summary = results.groupby(["model", "strategy"])[SUMMARY_METRICS].agg(["mean", "std"])
    summary[("fit_seconds", "total")] = results.groupby(["model", "strategy"]).fit_seconds.sum()
    summary.to_csv(out / "nested_cv_summary.csv")
    tag = f"niter{n_iter}"
    results.to_csv(res_dir / f"nested_cv_folds_{tag}.csv", index=False)
    summary.round(4).to_csv(res_dir / f"nested_cv_summary_{tag}.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary.round(4))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for (model, strategy), oof in oof_store.items():
        label = f"{model}/{strategy}"
        mp, fp = calibration_table(y, oof)
        a1.plot(mp, fp, marker="o", label=label)
        dca = decision_curve(y, oof)
        a2.plot(dca["thresholds"], dca["model"], label=label)
    a1.plot([0, 1], [0, 1], "k--")
    a1.set(title="Calibration (out-of-fold)", xlabel="Mean predicted", ylabel="Observed fraction")
    dca = decision_curve(y, next(iter(oof_store.values())))
    a2.plot(dca["thresholds"], dca["treat_all"], "k:", label="treat all")
    a2.plot(dca["thresholds"], dca["treat_none"], "k-", label="treat none")
    a2.set(title="Decision curve", xlabel="Threshold probability", ylabel="Net benefit",
           ylim=(-0.02, max(0.05, float(dca["treat_all"].max()))))
    a1.legend(fontsize=7)
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "calibration_dca.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--mode", choices=["nested", "temporal"], default="nested")
    ap.add_argument("--models", help="comma list, overrides config (e.g. xgboost)")
    ap.add_argument("--strategies", help="comma list, overrides config (e.g. none)")
    ap.add_argument("--n-iter", type=int, help="random-search iterations, overrides config")
    a = ap.parse_args()
    models = a.models.split(",") if a.models else None
    strategies = a.strategies.split(",") if a.strategies else None
    if a.mode == "temporal":
        cfg = yaml.safe_load(open(a.config))
        run_temporal(cfg, models or cfg["models"], strategies or cfg["imbalance"]["strategies"])
    else:
        main(a.config, models, strategies, a.n_iter)
