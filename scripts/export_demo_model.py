"""Exports the frozen primary model (logistic regression, set A, refit on 2011-2016 in the test fit
stage) to models/primary_logreg_A.json for the demo. Development data only; 2017-2018 is not read.

Checks, before writing:
  - the saved model's sha256 equals the one in results/test_2017_2018/run_log.json;
  - on every 2011-2016 row with complete numeric inputs, the portable predictions equal the
    sklearn model's (max absolute difference < 1e-9).

    python scripts/export_demo_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so `src` imports work

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from scripts.run_dev import load_dev  # noqa: E402
from scripts.run_test import MODELS, RUN_LOG, sha256  # noqa: E402
from src.models.portable import INPUTS, export, predict_proba  # noqa: E402

RUN = "logreg__none__A"
OUT = Path("models/primary_logreg_A.json")
TOL = 1e-9

if __name__ == "__main__":
    src = MODELS / f"{RUN}.joblib"
    log = json.loads(Path("results/test_2017_2018/run_log.json").read_text())
    want = log["models"][RUN]["model_sha256"]
    if sha256(src) != want:
        raise SystemExit(f"{src} does not match the run log ({RUN_LOG}); refusing to export")
    search = joblib.load(src)
    m = export(search)

    dev = load_dev(yaml.safe_load(open("configs/config.yaml")))
    X = dev[INPUTS]
    X = X[X[m["numeric"][:6]].notna().all(axis=1)]
    diff = np.abs(predict_proba(m, X) - search.predict_proba(X)[:, 1]).max()
    if diff >= TOL:
        raise SystemExit(f"portable predictions differ from the sklearn model by {diff:.2e}")

    m = {"name": "GlucoSense primary model: L2 logistic regression, set A, one-hot",
         "trained_on": "NHANES 2011-2016 (n = 13,189; 410 positives)",
         "outcome": "undiagnosed diabetes, HbA1c >= 6.5 % (main label)",
         "source_model_sha256": want, "settings_hash": log["models"][RUN]["settings_hash"],
         "pre_test_commit": log["git"]["commit"], "C": search.best_params_["model__C"],
         "check": {"rows": len(X), "max_abs_difference": float(diff)},
         "intended_use": "research and teaching only; not a medical device", **m}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(m, indent=1))
    print(f"wrote {OUT}: {len(X)} complete development rows, max |difference| {diff:.2e}")
