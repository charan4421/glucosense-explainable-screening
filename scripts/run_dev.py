"""Development-cycle analysis (2011-2016 only), as fixed in docs/analysis_plan.md.

Stages, run in this order (cheapest first, so a crash loses the least work):
  main    main models x sets A, B, C (strategy none)
  d       main models on set D = best of B/C (rule) + healthcare access; sensitivity only
  loco    leave-one-cycle-out, main models on the primary set (rule)
  extras  set A only: rebalancing variants and TabNet
Then: summary (mean +/- SD), selection rules, and paired model/set differences on identical folds
with Nadeau-Bengio corrected 95 % intervals (results/dev_cv_v2_paired.csv).

Every combination is checkpointed to results/dev_cv_v2/ (LOCO: results/dev_cv_v2/loco/) with a
hash of its data, settings and features, and skipped on restart when the hash matches.

    python scripts/run_dev.py                 # all stages
    python scripts/run_dev.py --stage main --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so `src` imports work

import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import yaml  # noqa: E402

from src.data.nhanes import MAIN_LABEL  # noqa: E402
from src.evaluation.checkpoint import (  # noqa: E402
    build_summary,
    combo_hash,
    combo_path,
    is_done,
    save_combo,
    save_oof,
)
from src.evaluation.compare import comparison_table  # noqa: E402
from src.evaluation.loco import loco  # noqa: E402
from src.evaluation.nested_cv import nested_cv  # noqa: E402
from src.evaluation.selection import base_for_d, primary_model, primary_set  # noqa: E402
from src.evaluation.temporal import feature_set  # noqa: E402

OUT = Path("results/dev_cv_v2")
LOCO_DIR = OUT / "loco"
OOF_DIR = Path("reports/dev_cv_v2_oof")
SUMMARY = Path("results/dev_cv_v2_summary.csv")
T0 = time.perf_counter()

# KNN imputation computes distances in chunks; cap the chunk memory (default 1 GB) for 8 GB RAM.
sklearn.set_config(working_memory=128)


def load_dev(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    df = pd.read_parquet(d["dev_processed_path"])
    assert d["temporal_test_cycle"] not in set(df.cycle), "test cycle in the development data"
    assert set(df.cycle) == set(d["dev_cycles"]), sorted(df.cycle.unique())
    return df.reset_index(drop=True)


def run_combo(cfg, df, model, strategy, set_name, n_iter, base=None, dry=False, stage="main"):
    feats, cat = feature_set(cfg, set_name, base)
    h = combo_hash(cfg, feats, cat, n_iter, cfg["data"]["dev_processed_path"])
    out_dir = LOCO_DIR if stage == "loco" else OUT
    path = combo_path(out_dir, model, strategy, set_name)
    tag = f"[{stage}] {model} / {strategy} / set {set_name} ({len(feats)} features)"
    if is_done(path, h):
        print(f"{tag}: checkpoint matches, skipped", flush=True)
        return
    if dry:
        print(f"{tag}: would run", flush=True)
        return
    print(f"{tag}: running", flush=True)
    t0 = time.perf_counter()
    if stage == "loco":
        res = loco(df, feats, cat, MAIN_LABEL, model, strategy, cfg, n_iter)
    else:
        cv = cfg["cv"]
        res, oof = nested_cv(df[feats], df[MAIN_LABEL], model, strategy, cat, cv["outer_folds"],
                             cv["inner_folds"], n_iter, cv["scoring"],
                             cfg["preprocessing"]["knn_neighbors"],
                             cfg["features"]["interactions"], cfg["seed"],
                             cv.get("outer_repeats", 1))
        save_oof(oof, OOF_DIR / f"{model}__{strategy}__{set_name}.csv", h)
    res.insert(0, "feature_set", set_name)
    if base:
        res.insert(1, "d_base", base)
    save_combo(res, path, h)
    print(f"{tag}: done in {time.perf_counter() - t0:.0f}s (total elapsed "
          f"{time.perf_counter() - T0:.0f}s)  PR-AUC {res.pr_auc.mean():.4f}", flush=True)


def main(cfg_path: str, stages: list[str], dry: bool) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    df, des = load_dev(cfg), cfg["design"]
    n_iter = cfg["cv"]["n_iter"]
    print(f"development cycles {sorted(df.cycle.unique())}, n={len(df)}, "
          f"positives={int(df[MAIN_LABEL].sum())}; {cfg['data']['temporal_test_cycle']} not loaded",
          flush=True)
    order = sorted(des["main_models"], key=["logreg", "xgboost", "mlp", "ebm"].index)
    selection = {}

    if "main" in stages:
        for model in order:
            for s in des["main_sets"]:
                run_combo(cfg, df, model, "none", s, n_iter, dry=dry)
    if dry:
        return
    summary = build_summary(OUT, SUMMARY)

    if "d" in stages:
        base = base_for_d(summary, des["main_models"])
        selection["d_base"] = base
        for model in order:
            run_combo(cfg, df, model, "none", "D", n_iter, base=base)
        summary = build_summary(OUT, SUMMARY)

    if "loco" in stages:
        ps = primary_set(summary, des["main_models"])
        selection.update(ps)
        for model in order:
            run_combo(cfg, df, model, "none", ps["primary_set"], n_iter, stage="loco")

    if "extras" in stages:
        for model, strategy in des["set_a_only"]:
            run_combo(cfg, df, model, strategy, "A", n_iter)
        summary = build_summary(OUT, SUMMARY)

    ps = primary_set(summary, des["main_models"])
    pm = primary_model(summary, ps["primary_set"], des["simplicity_order"])
    selection.update({**ps, **pm})
    Path("results/dev_cv_v2_selection.json").write_text(
        json.dumps({k: (float(v) if hasattr(v, "item") else v) for k, v in selection.items()},
                   indent=2))
    folds = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("*__*.csv"))], ignore_index=True)
    sets = [s for s in ["A", "B", "C", "D"] if s in set(folds.feature_set)]
    comparison_table(folds, des["main_models"], sets, cfg["cv"]["outer_folds"]).to_csv(
        "results/dev_cv_v2_paired.csv", index=False, float_format="%.5f")
    if LOCO_DIR.exists() and any(LOCO_DIR.glob("*.csv")):
        pd.concat([pd.read_csv(f) for f in sorted(LOCO_DIR.glob("*.csv"))]).to_csv(
            "results/dev_cv_v2_loco.csv", index=False, float_format="%.4f")
    print(json.dumps(selection, indent=2, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--stage", default="main,d,loco,extras")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(a.config, a.stage.split(","), a.dry_run)
