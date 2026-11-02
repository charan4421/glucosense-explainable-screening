"""Exploratory, development data only (2011-2016): primary model by sex from the nested-CV
out-of-fold predictions, with the women - men difference. Never loads 2017-2018.

Same bootstrap as scripts/post_test_exploratory.py (within each sex, 1000 resamples, seed 42), so
the development and test gaps are computed the same way.

    python scripts/dev_sex_gap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so `src` imports work

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from scripts.dev_subgroups import load_dev_oof, primary_run  # noqa: E402
from scripts.post_test_exploratory import PRIMARY, by_sex  # noqa: E402
from src.data.nhanes import MAIN_LABEL  # noqa: E402

OUT = Path("results/dev_cv_v2_sex_gap")
LABEL = "development out-of-fold, exploratory"

if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/config.yaml"))
    run = primary_run()
    dev, oof = load_dev_oof(cfg, run)
    df = pd.DataFrame({"y": dev[MAIN_LABEL].to_numpy(), "sex": dev.sex.to_numpy(), PRIMARY: oof})
    OUT.mkdir(parents=True, exist_ok=True)
    groups, diffs, _ = by_sex(df)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        for name, tab in {"sex_groups.csv": groups, "sex_difference.csv": diffs}.items():
            tab = tab.assign(label=LABEL, run=run)
            tab.to_csv(OUT / name, index=False, float_format="%.5f")
            print(f"\n{name} ({run}, {LABEL})\n"
                  f"{tab.drop(columns=['label', 'run']).round(4).to_string(index=False)}")
