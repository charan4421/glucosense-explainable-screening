"""Rebuild results/dev_cv_summary.csv from whatever combination files exist (safe to run any time).

    python scripts/summarize_dev_cv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so `src` imports work

import pandas as pd  # noqa: E402

from src.evaluation.checkpoint import build_summary  # noqa: E402

if __name__ == "__main__":
    s = build_summary(Path("results/dev_cv"), Path("results/dev_cv_summary.csv"))
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(s.round(4))
