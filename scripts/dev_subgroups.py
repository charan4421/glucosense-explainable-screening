"""Subgroup performance (sex, race/ethnicity) from saved out-of-fold predictions, 2011-2016 only.

    python scripts/dev_subgroups.py --model xgboost --strategy none
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so `src` imports work

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.data.nhanes import MAIN_LABEL  # noqa: E402
from src.evaluation.checkpoint import load_oof, run_hash  # noqa: E402
from src.evaluation.subgroups import RACE_ETH, subgroup_table  # noqa: E402
from src.evaluation.temporal import temporal_split  # noqa: E402


def load_dev_oof(cfg: dict, model: str, strategy: str, n_iter: int):
    """Development frame (same row order as the nested CV) and its out-of-fold predictions."""
    test_cycle = cfg["data"]["temporal_test_cycle"]
    dev, _ = temporal_split(pd.read_parquet(cfg["data"]["processed_path"]), test_cycle)
    assert test_cycle not in set(dev.cycle), "test cycle leaked into the development analysis"
    dev = dev.reset_index(drop=True)
    path = Path("reports/dev_cv_oof") / f"{model}__{strategy}.csv"
    oof = load_oof(path, run_hash(cfg, cfg["seed"], n_iter))
    if oof is None or len(oof) != len(dev):
        raise SystemExit(f"{path} missing, stale or wrong length: rerun the nested CV first")
    return dev, oof


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model", default="xgboost")
    ap.add_argument("--strategy", default="none")
    ap.add_argument("--n-iter", type=int, default=5)
    ap.add_argument("--min-pos", type=int, default=30)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    dev, oof = load_dev_oof(cfg, a.model, a.strategy, a.n_iter)
    tab = subgroup_table(dev[MAIN_LABEL], oof,
                         {"sex": dev.sex, "race_eth": dev.race_eth.map(RACE_ETH)}, a.min_pos)
    out = Path("results") / f"subgroups_{a.model}__{a.strategy}.csv"
    tab.to_csv(out, index=False, float_format="%.4f")
    print(f"{a.model}/{a.strategy}, out-of-fold, 2011-2016 (n={len(dev)}) -> {out}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tab.round(4).to_string(index=False))
