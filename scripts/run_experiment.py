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
from src.evaluation.metrics import calibration_table, decision_curve  # noqa: E402
from src.evaluation.nested_cv import nested_cv  # noqa: E402
from src.evaluation.temporal import feature_columns, temporal_evaluate  # noqa: E402

TARGET = MAIN_LABEL


def run_temporal(cfg: dict, models: list[str], strategies: list[str]) -> pd.DataFrame:
    out = Path("reports")
    out.mkdir(exist_ok=True)
    df = pd.read_parquet(cfg["data"]["processed_path"])
    feats, cat = feature_columns(cfg), cfg["features"]["categorical"]
    test_cycle = cfg["data"]["temporal_test_cycle"]
    rows = []
    for model in models:
        for strategy in strategies:
            print(f"== temporal {model} / {strategy}: test {test_cycle}")
            rows.append(temporal_evaluate(df, feats, cat, TARGET, test_cycle, model, strategy, cfg))
    res = pd.DataFrame(rows)
    res.to_csv(out / "temporal_test.csv", index=False)
    cols = ["model", "strategy", "n_train", "n_test", "train_prevalence", "test_prevalence",
            "pr_auc", "roc_auc", "brier"]
    with pd.option_context("display.width", 160):
        print(res[cols].round(4).to_string(index=False))
    for r in rows:   # PR-AUC is only meaningful against the base rate it has to beat
        print(f"{r['model']}/{r['strategy']}: PR-AUC {r['pr_auc']:.4f} vs test prevalence "
              f"{r['test_prevalence']:.4f} ({r['pr_auc'] / r['test_prevalence']:.1f}x the base rate)")
    return res


def main(cfg_path: str, models: list[str] | None = None, strategies: list[str] | None = None) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    out = Path("reports")
    out.mkdir(exist_ok=True)

    df = pd.read_parquet(cfg["data"]["processed_path"])
    feats = feature_columns(cfg)
    X, y = df[feats], df[TARGET]
    cat = cfg["features"]["categorical"]
    cv = cfg["cv"]

    all_rows, oof_store = [], {}
    for model in models or cfg["models"]:
        for strategy in strategies or cfg["imbalance"]["strategies"]:
            if model == "tabnet" and strategy == "class_weight":
                continue
            print(f"== {model} / {strategy}")
            res, oof = nested_cv(
                X, y, model, strategy, cat, cv["outer_folds"], cv["inner_folds"], cv["n_iter"],
                cv["scoring"], cfg["preprocessing"]["knn_neighbors"],
                cfg["features"]["interactions"], cfg["seed"])
            all_rows.append(res)
            oof_store[(model, strategy)] = oof

    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(out / "nested_cv_folds.csv", index=False)
    summary = (results.groupby(["model", "strategy"])[["pr_auc", "roc_auc", "brier"]]
               .agg(["mean", "std"]).round(4))
    summary.to_csv(out / "nested_cv_summary.csv")
    print(summary)

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
    a = ap.parse_args()
    models = a.models.split(",") if a.models else None
    strategies = a.strategies.split(",") if a.strategies else None
    if a.mode == "temporal":
        cfg = yaml.safe_load(open(a.config))
        run_temporal(cfg, models or cfg["models"], strategies or cfg["imbalance"]["strategies"])
    else:
        main(a.config, models, strategies)
