"""Skip patterns: structural "no" answers become real values before imputation."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.derive import RAW_COLUMNS, STRUCTURAL, assert_no_structural_missing, derive
from src.data.nhanes import fix_xpt_zero

NA = np.nan


def raw_row(**over):
    """A 50-year-old man who answers 'no' to every gate, so every follow-up is NaN (as NHANES)."""
    row = {c: NA for cols in RAW_COLUMNS.values() for c in cols}
    row.update(RIAGENDR=1, RIDAGEYR=50, SMQ020=2, SLD010H=7, SLQ050=2,
               PAQ605=2, PAQ620=2, PAQ635=2, PAQ650=2, PAQ665=2, PAD680=300,
               ALQ101=2, ALQ110=2, DBQ700=3, DBD895=0, DBD905=0, DBD910=0,
               BPQ020=2, WHD020=180, WHD050=180, WHD110=170, WHD120=160, WHD140=190, WHQ070=2)
    row.update(over)
    return row


def frame(*rows):
    return pd.DataFrame(list(rows))


def test_structural_no_becomes_zero():
    d = derive(frame(raw_row()))
    for col in ["pa_met_min_week", "cigs_per_day", "drinks_per_week", "binge_days_per_year",
                "fast_food_week", "htn_med_now", "gdm_history", "baby_9lb"]:
        assert d.loc[0, col] == 0, col
    assert d.loc[0, "smoking_status"] == "never"
    assert d.loc[0, "drinking_status"] == "lifetime_abstainer"


def test_not_applicable_keeps_indicator():
    young = derive(frame(raw_row(RIDAGEYR=24)))          # too young for weight at 25 / 10 y ago
    assert young.loc[0, "weight_change_10y_kg"] == 0 and young.loc[0, "weight_change_10y_kg_na"] == 1
    assert young.loc[0, "weight_gain_since_25_kg"] == 0
    assert young.loc[0, "weight_gain_since_25_kg_na"] == 1
    old = derive(frame(raw_row()))
    assert old.loc[0, "weight_change_10y_kg_na"] == 0
    assert np.isclose(old.loc[0, "weight_change_10y_kg"], 10 * 0.45359237)


def test_answered_values_and_refusals():
    d = derive(frame(
        raw_row(PAQ650=1, PAQ655=3, PAD660=60),              # 3 x 60 min vigorous -> 8 * 180
        raw_row(PAQ650=1, PAQ655=99, PAD660=60),             # don't know -> NaN (imputed later)
        raw_row(SMQ020=1, SMQ040=1, SMD641=30, SMD650=10),   # current smoker, 10 a day
        raw_row(ALQ101=1, ALQ120Q=2, ALQ120U=1, ALQ130=3, ALQ141Q=0),  # 2 days/week x 3
        raw_row(ALQ101=1, ALQ120Q=0),                        # drank before, not in past year
        raw_row(RIAGENDR=2, RHQ131=1, RHQ162=1, RHQ172=2),
        raw_row(RIAGENDR=2, RHQ131=1, RHQ162=3, RHQ172=1),   # borderline GDM -> 0
    ))
    assert d.pa_met_min_week[0] == 8 * 180 and np.isnan(d.pa_met_min_week[1])
    assert d.cigs_per_day[2] == 10 and d.smoking_status[2] == "current"
    assert d.drinks_per_week[3] == 6 and d.binge_days_per_year[3] == 0
    assert d.drinks_per_week[4] == 0 and d.drinking_status[4] == "former"
    assert d.gdm_history[5] == 1 and d.baby_9lb[5] == 0
    assert d.gdm_history[6] == 0 and d.baby_9lb[6] == 1


def test_guard_fails_when_structural_missing_reaches_imputer():
    raw = frame(raw_row(), raw_row(RIDAGEYR=24), raw_row(RIAGENDR=2, RHQ131=2))
    d = derive(raw)
    assert_no_structural_missing(raw, d)                    # correct derivation passes
    for col in STRUCTURAL:
        broken = d.copy()
        broken[col] = np.nan                                # e.g. a gate 'no' left as NaN
        with pytest.raises(ValueError, match=col):
            assert_no_structural_missing(raw, broken)


def test_xpt_zero_is_restored():
    df = fix_xpt_zero(pd.DataFrame({"ALQ120Q": [5.397605346934028e-79, 2.0, NA]}))
    assert df.ALQ120Q.tolist()[:2] == [0.0, 2.0] and np.isnan(df.ALQ120Q[2])


DEV = Path(yaml.safe_load(open("configs/config.yaml"))["data"]["dev_processed_path"])


@pytest.mark.skipif(not DEV.exists(), reason="development data not built")
def test_built_data_has_no_structural_missing_in_features():
    """On the real build: every skip-pattern feature is filled wherever its gate fixes it."""
    df = pd.read_parquet(DEV)
    raw = df[[c for c in df if c.startswith("gate_")]].rename(columns=lambda c: c[5:])
    assert_no_structural_missing(raw, df)
    assert "2017-2018" not in set(df.cycle)
