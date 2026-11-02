"""One-command demo: score one patient with the frozen primary model (models/primary_logreg_A.json).

Research and teaching only; not a medical device and not a diagnosis.

    python scripts/demo.py                                  # the fictional sample patient
    python scripts/demo.py --patient my_patient.json

The patient file holds the 10 set-A inputs (see examples/sample_patient.json). race_eth uses the
NHANES RIDRETH3 codes: 1 Mexican American, 2 Other Hispanic, 3 Non-Hispanic White,
4 Non-Hispanic Black, 6 Non-Hispanic Asian, 7 Other / multiracial.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so `src` imports work

import pandas as pd  # noqa: E402

from src.models.portable import INPUTS, load, predict_proba  # noqa: E402

MODEL = Path("models/primary_logreg_A.json")
SAMPLE = Path("examples/sample_patient.json")


def score(patient: dict, model_path: Path = MODEL) -> float:
    missing = [c for c in INPUTS if c not in patient]
    if missing:
        raise ValueError(f"missing inputs: {missing}")
    row = pd.DataFrame([{c: patient[c] for c in INPUTS}])
    row["race_eth"] = row["race_eth"].astype(str)
    return float(predict_proba(load(model_path), row)[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default=str(SAMPLE))
    a = ap.parse_args()
    patient = json.loads(Path(a.patient).read_text())
    p = score(patient)
    print(json.dumps({c: patient[c] for c in INPUTS}))
    print(f"Estimated probability of undiagnosed diabetes (HbA1c >= 6.5 %): {p:.1%}")
    print("Research demo only: not a diagnosis. The 2017-2018 test calibrated less well in men; "
          "see docs/model_card.md.")
