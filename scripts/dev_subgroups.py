"""Subgroup performance (sex, race/ethnicity) from saved out-of-fold predictions, 2011-2016 only,
with percentile bootstrap 95 % intervals resampled within each group (docs/analysis_plan.md,
deviation 2026-11-02).

    python scripts/dev_subgroups.py                      # primary model on the primary set
    python scripts/dev_subgroups.py --run ebm__none__C
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so `src` imports work

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from scripts.run_dev import OOF_DIR, load_dev  # noqa: E402
from src.data.nhanes import MAIN_LABEL  # noqa: E402
from src.evaluation.checkpoint import combo_hash, load_oof  # noqa: E402
from src.evaluation.subgroups import RACE_ETH, subgroup_table  # noqa: E402
from src.evaluation.temporal import feature_set  # noqa: E402

SELECTION = Path("results/dev_cv_v2_selection.json")


def primary_run() -> str:
    s = json.loads(SELECTION.read_text())
    return f"{s['primary_model']}__none__{s['primary_set']}"


def load_dev_oof(cfg: dict, run: str):
    """Development frame (same row order as the nested CV) and the out-of-fold predictions of
    run = '<model>__<strategy>__<set>', checked against the combination hash."""
    model, strategy, set_name = run.split("__")
    base = json.loads(SELECTION.read_text())["d_base"] if set_name == "D" else None
    feats, cat = feature_set(cfg, set_name, base)
    h = combo_hash(cfg, feats, cat, cfg["cv"]["n_iter"], cfg["data"]["dev_processed_path"])
    dev = load_dev(cfg)
    path = OOF_DIR / f"{run}.csv"
    oof = load_oof(path, h)
    if oof is None or len(oof) != len(dev):
        raise SystemExit(f"{path} missing, stale or wrong length: rerun scripts/run_dev.py first")
    return dev, oof


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--run", default=None, help="<model>__<strategy>__<set>; default: primary")
    ap.add_argument("--min-pos", type=int, default=30)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    run = a.run or primary_run()
    dev, oof = load_dev_oof(cfg, run)
    tab = subgroup_table(dev[MAIN_LABEL], oof,
                         {"sex": dev.sex, "race_eth": dev.race_eth.map(RACE_ETH)}, a.min_pos,
                         n_boot=a.n_boot, seed=cfg["seed"])
    out = Path("results") / f"dev_cv_v2_subgroups_{run}.csv"
    tab.to_csv(out, index=False, float_format="%.4f")
    print(f"{run}, out-of-fold, 2011-2016 (n={len(dev)}), {a.n_boot} bootstrap resamples -> {out}")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(tab.round(4).to_string(index=False))
