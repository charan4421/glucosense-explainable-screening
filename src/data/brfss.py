"""BRFSS loader for external validation (overlapping features only).

Download the BRFSS annual data (e.g. 2022 SAS transport / CSV) from
https://www.cdc.gov/brfss/ and point `load_brfss` at the CSV. Column names below follow the
2022 codebook; check them against your year's codebook before running.

BRFSS has no labs. 'Positive' here means self-reported diabetes (DIABETE4 == 1), which is a
different target from NHANES undiagnosed diabetes. Say so in the paper: this is a transfer check
of ranking/calibration on a related outcome, not a like-for-like replication.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLS = ["_AGE80", "_SEX", "_BMI5", "DIABETE4", "SMOKE100"]


def load_brfss(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, usecols=COLS)
    df = pd.DataFrame({
        "age": raw["_AGE80"],
        "sex": raw["_SEX"].map({1: "male", 2: "female"}),
        "bmi": raw["_BMI5"] / 100.0,        # BRFSS stores BMI x100
        "smoker": raw["SMOKE100"].map({1: "yes", 2: "no"}),
    })
    df["diabetes"] = (raw["DIABETE4"] == 1).astype(int)
    df = df[raw["DIABETE4"].isin([1, 2, 3, 4])]
    return df.replace({np.inf: np.nan}).reset_index(drop=True)
