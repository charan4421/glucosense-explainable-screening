"""Checkpoint/resume for the development-cycle nested CV.

Each model/strategy writes results/dev_cv/<model>__<strategy>.csv as soon as it finishes, one
row per outer fold, with a run_hash column. A restart skips a combination only when its file
exists and the stored hash matches the current config + seed + n_iter; anything else reruns.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FOLD_COLUMNS = ["pr_auc", "roc_auc", "brier", "calibration_slope", "calibration_intercept",
                "citl_difference", "citl_ratio_observed_expected", "n_pos", "n", "fit_seconds"]


def run_hash(cfg: dict, seed: int, n_iter: int) -> str:
    """Stable hash of the config, seed and n_iter (key order does not matter)."""
    blob = json.dumps({"cfg": cfg, "seed": seed, "n_iter": n_iter}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def combo_hash(cfg: dict, feats: list[str], categorical: list[str], n_iter: int,
               data_path: str | Path, encoding: str = "ordinal") -> str:
    """Hash of everything that changes one combination's result: data file content, sample/CV/
    preprocessing settings, the resolved feature list, seed and n_iter. Unlike run_hash it does
    not depend on unrelated config (other sets, the design list), so adding work elsewhere
    never invalidates finished combinations. A non-default categorical encoding is part of
    the hash; the default (ordinal) is left out so existing hashes stay valid."""
    digest = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    relevant = {"data_sha256": digest, "cv": cfg["cv"], "preprocessing": cfg["preprocessing"],
                "interactions": cfg["features"]["interactions"], "features": list(feats),
                "categorical": list(categorical)}
    if encoding != "ordinal":
        relevant["categorical_encoding"] = encoding
    return run_hash(relevant, cfg["seed"], n_iter)


def combo_path(out_dir: Path, model: str, strategy: str, feature_set: str | None = None) -> Path:
    name = f"{model}__{strategy}" + (f"__{feature_set}" if feature_set else "")
    return Path(out_dir) / f"{name}.csv"


def is_done(path: Path, h: str) -> bool:
    """True only if the file exists, is readable, has rows, and every row carries hash h."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, dtype={"run_hash": str})
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return False
    return len(df) > 0 and "run_hash" in df and bool((df.run_hash == h).all())


def save_combo(folds: pd.DataFrame, path: Path, h: str) -> None:
    """Write via a temp file and rename, so a crash mid-write never leaves a file that passes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    folds.assign(run_hash=h).to_csv(tmp, index=False)
    tmp.replace(path)


def save_oof(oof: np.ndarray, path: Path, h: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"p": oof, "run_hash": h}).to_csv(path, index=False)


def load_oof(path: Path, h: str) -> np.ndarray | None:
    """Out-of-fold predictions for the calibration/DCA plot, or None if missing or stale."""
    return pd.read_csv(path).p.to_numpy() if is_done(path, h) else None


def build_summary(in_dir: Path, out_path: Path) -> pd.DataFrame:
    """Mean and SD across outer folds per model/strategy, from every combination file in in_dir."""
    files = sorted(Path(in_dir).glob("*__*.csv"))
    if not files:
        raise FileNotFoundError(f"no combination files in {in_dir}")
    folds = pd.concat([pd.read_csv(f, dtype={"run_hash": str}) for f in files], ignore_index=True)
    keys = [k for k in ("feature_set", "model", "strategy") if k in folds]
    g = folds.groupby(keys, sort=False)
    stats = g[FOLD_COLUMNS].agg(["mean", "std"])
    stats.columns = [f"{c}_{'sd' if s == 'std' else s}" for c, s in stats.columns]
    meta = g.agg(n_folds=("fold", "size"), n_iter=("n_iter", "first"),
                 run_hash=("run_hash", "first"), fit_seconds_total=("fit_seconds", "sum"))
    summary = meta.join(stats).reset_index()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False, float_format="%.4f")
    return summary
