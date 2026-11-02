"""Calibration across race/ethnicity, exploratory extension (docs/analysis_plan.md, deviation
2026-11-02 A). Logistic regression on set A, same nested CV and folds as the main grid,
2011-2016 only. The primary model is unchanged whatever the result.

    python scripts/dev_calib_ext.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so `src` imports work

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from scripts.dev_subgroups import load_dev_oof  # noqa: E402
from scripts.run_dev import load_dev  # noqa: E402
from src.data.nhanes import MAIN_LABEL  # noqa: E402
from src.evaluation.calibration_ext import decide, group_recalibration, race_pipeline  # noqa: E402
from src.evaluation.checkpoint import FOLD_COLUMNS  # noqa: E402
from src.evaluation.nested_cv import nested_cv  # noqa: E402
from src.evaluation.subgroups import RACE_ETH, subgroup_table  # noqa: E402
from src.evaluation.temporal import feature_set  # noqa: E402

OUT = Path("results/dev_cv_v2_calib_ext")
PRIMARY = "logreg__none__A"
CANDIDATES = ["onehot_interactions", "group_recalibration"]   # decision rule applies to these
DESCRIPTIVE = ["onehot_main"]                                  # encoding alone; not a candidate


def summarise(folds: pd.DataFrame, name: str) -> dict:
    out = {"run": name, "n_folds": len(folds)}
    for c in FOLD_COLUMNS[:7]:
        out[f"{c}_mean"], out[f"{c}_sd"] = folds[c].mean(), folds[c].std()
    return out


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/config.yaml"))
    cv, seed = cfg["cv"], cfg["seed"]
    df = load_dev(cfg)
    feats, cat = feature_set(cfg, "A")
    knn, inter = cfg["preprocessing"]["knn_neighbors"], cfg["features"]["interactions"]
    groups = {"sex": df.sex, "race_eth": df.race_eth.map(RACE_ETH)}
    OUT.mkdir(parents=True, exist_ok=True)

    offsets: list[dict] = []
    hooks = {
        "onehot_interactions": {"make_pipeline": race_pipeline(cat, True, knn, inter, seed)},
        "group_recalibration": {"postprocess": group_recalibration(log=offsets)},
        "onehot_main": {"make_pipeline": race_pipeline(cat, False, knn, inter, seed)},
    }
    primary_folds = pd.read_csv(Path("results/dev_cv_v2") / f"{PRIMARY}.csv")
    _, primary_oof = load_dev_oof(cfg, PRIMARY)
    subs = {PRIMARY: subgroup_table(df[MAIN_LABEL], primary_oof, groups, n_boot=1000, seed=seed)}
    summary = [summarise(primary_folds, PRIMARY)]
    for name, hook in hooks.items():
        t0 = time.perf_counter()
        folds, oof = nested_cv(df[feats], df[MAIN_LABEL], "logreg", "none", cat,
                               cv["outer_folds"], cv["inner_folds"], cv["n_iter"], cv["scoring"],
                               knn, inter, seed, cv["outer_repeats"], **hook)
        folds.insert(0, "run", name)
        folds.to_csv(OUT / f"{name}_folds.csv", index=False)
        pd.DataFrame({"p": oof}).to_csv(OUT / f"{name}_oof.csv", index=False)
        subs[name] = subgroup_table(df[MAIN_LABEL], oof, groups, n_boot=1000, seed=seed)
        summary.append(summarise(folds, name))
        print(f"{name}: PR-AUC {folds.pr_auc.mean():.4f} +/- {folds.pr_auc.std():.4f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    pd.DataFrame(offsets).rename(columns=RACE_ETH).rename_axis("outer_fold").to_csv(
        OUT / "group_recalibration_offsets.csv", float_format="%.4f")
    summary = pd.DataFrame(summary)
    summary.to_csv("results/dev_cv_v2_calib_ext_summary.csv", index=False, float_format="%.4f")
    pd.concat([s.assign(run=n) for n, s in subs.items()]).to_csv(
        "results/dev_cv_v2_calib_ext_subgroups.csv", index=False, float_format="%.4f")

    s = summary.set_index("run")
    decision = decide(subs[PRIMARY], (s.loc[PRIMARY, "pr_auc_mean"], s.loc[PRIMARY, "pr_auc_sd"]),
                      {n: (subs[n], s.loc[n, "pr_auc_mean"]) for n in CANDIDATES})
    decision["descriptive"] = {n: decide(subs[PRIMARY], (s.loc[PRIMARY, "pr_auc_mean"],
                                                         s.loc[PRIMARY, "pr_auc_sd"]),
                                         {n: (subs[n], s.loc[n, "pr_auc_mean"])})["candidates"][n]
                               for n in DESCRIPTIVE}
    decision["label"] = ("exploratory extension; primary model stays logistic regression on set A; "
                         "not applied to 2017-2018 without a new deviation")
    Path("results/dev_cv_v2_calib_ext_decision.json").write_text(
        json.dumps(decision, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    print(json.dumps(decision, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
