"""Run the full comparison: models x imbalance strategies under nested CV, plus temporal test."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.evaluation.metrics import calibration_table, decision_curve  # noqa: E402
from src.evaluation.nested_cv import nested_cv  # noqa: E402

TARGET = "undiagnosed_diabetes"


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    out = Path("reports")
    out.mkdir(exist_ok=True)

    df = pd.read_parquet(cfg["data"]["processed_path"])
    feats = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    X, y = df[feats], df[TARGET]
    cat = cfg["features"]["categorical"]
    cv = cfg["cv"]

    all_rows, oof_store = [], {}
    for model in cfg["models"]:
        for strategy in cfg["imbalance"]["strategies"]:
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
    main(ap.parse_args().config)
