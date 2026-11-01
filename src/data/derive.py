"""Derived non-lab features (feature sets B, C and D) from raw NHANES 2011-2016 columns.

Codes follow the CDC codebooks (see docs/feature_audit.md). Refused / don't know become NaN, which
the pipeline imputes. Structural skips do NOT stay NaN: a "no" to a gate question means the
follow-up quantity is a real zero (0 minutes, 0 cigarettes, 0 drinks), and it is set before
imputation. Where zero would mean something different from "not asked", a *_na indicator is kept.
`assert_no_structural_missing` enforces this on every build.

Cross-cycle harmonisation (2011-2016):
- sleep hours: SLD010H (2011-2014) / SLD012 (2015-2016), capped at 12 (2011-2014 top code).
- drinks per drinking day (ALQ130) capped at 15 (2015-2016 top code "15 or more").
- ready-to-eat / frozen meals per 30 days (DBD905/910) capped at 90 (2015-2016 code 6666 = > 90).
- healthcare visits: HUQ050 (2011-2012) / HUQ051 (2013+) mapped to common bins
  0, 1, 2-3, 4-9, 10-12, 13+.
- gestational diabetes "borderline" (RHQ162 = 3) is coded 0: the ADA item asks about a diagnosis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LB_TO_KG = 0.45359237

# Raw columns needed per file (2011-2016 names).
RAW_COLUMNS = {
    "DEMO": ["RIDEXPRG", "DMDEDUC2", "INDFMPIR"],
    "PAQ": ["PAQ605", "PAQ610", "PAD615", "PAQ620", "PAQ625", "PAD630", "PAQ635", "PAQ640",
            "PAD645", "PAQ650", "PAQ655", "PAD660", "PAQ665", "PAQ670", "PAD675", "PAD680"],
    "SLQ": ["SLD010H", "SLD012", "SLQ050"],
    "SMQ": ["SMQ040", "SMD641", "SMD650"],
    "ALQ": ["ALQ101", "ALQ110", "ALQ120Q", "ALQ120U", "ALQ130", "ALQ141Q", "ALQ141U"],
    "DBQ": ["DBQ700", "DBD895", "DBD900", "DBD905", "DBD910"],
    "DR1TOT": ["DR1DRSTZ", "DR1TKCAL", "DR1TCARB", "DR1TSUGR", "DR1TFIBE", "DR1TSFAT"],
    "BPQ": ["BPQ020", "BPQ040A", "BPQ050A"],
    "RHQ": ["RHQ131", "RHQ162", "RHQ172"],
    "WHQ": ["WHD020", "WHD050", "WHD110", "WHD120", "WHD140", "WHQ070"],
    "FSQ": ["FSDAD"],
    "HIQ": ["HIQ011"],
    "HUQ": ["HUQ030", "HUQ050", "HUQ051", "HUQ071"],
}

# GPAQ domains: (gate yes/no, days per week, minutes per day, MET weight)
PA_DOMAINS = {
    "work_vigorous": ("PAQ605", "PAQ610", "PAD615", 8.0),
    "work_moderate": ("PAQ620", "PAQ625", "PAD630", 4.0),
    "transport": ("PAQ635", "PAQ640", "PAD645", 4.0),
    "rec_vigorous": ("PAQ650", "PAQ655", "PAD660", 8.0),
    "rec_moderate": ("PAQ665", "PAQ670", "PAD675", 4.0),
}
HUQ051_TO_HUQ050 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 4, 7: 5, 8: 5}


def _col(raw: pd.DataFrame, name: str) -> pd.Series:
    return raw[name] if name in raw else pd.Series(np.nan, index=raw.index)


def _na(s: pd.Series, codes) -> pd.Series:
    return s.where(~s.isin(codes))


def _yes_no(s: pd.Series) -> pd.Series:
    return s.map({1.0: 1.0, 2.0: 0.0})


def _categorical(conditions: list[pd.Series], labels: list[str]) -> pd.Series:
    """First matching label per row; NaN where no condition holds."""
    out = pd.Series(np.nan, index=conditions[0].index, dtype=object)
    for cond, label in reversed(list(zip(conditions, labels, strict=True))):
        out[cond.to_numpy()] = label
    return out


def _per_week(q: pd.Series, unit: pd.Series) -> pd.Series:
    return q * unit.map({1.0: 1.0, 2.0: 12 / 52, 3.0: 1 / 52})


def derive(raw: pd.DataFrame) -> pd.DataFrame:
    """Feature columns for sets B, C, D. `raw` is one or more cycles of merged raw columns
    (all rows in the analysis sample), with RIAGENDR and RIDAGEYR."""
    r, out = raw, pd.DataFrame(index=raw.index)
    female, age = r.RIAGENDR == 2, r.RIDAGEYR

    # ---- physical activity: MET-minutes per week (GPAQ weights), sedentary minutes per day
    met = pd.Series(0.0, index=r.index)
    for gate, days, mins, w in PA_DOMAINS.values():
        g = r[gate]
        dur = _na(r[days], [77, 99]) * _na(r[mins], [7777, 9999])
        dom = pd.Series(np.where(g == 2, 0.0, np.where(g == 1, dur, np.nan)), index=r.index)
        met = met + w * dom                              # NaN in any domain -> NaN total
    out["pa_met_min_week"] = met
    out["sedentary_min_day"] = _na(r.PAD680, [7777, 9999])

    # ---- sleep
    hours = _na(_col(r, "SLD010H"), [77, 99]).fillna(_col(r, "SLD012"))
    out["sleep_hours"] = hours.clip(upper=12)
    out["sleep_trouble_told"] = _yes_no(r.SLQ050)

    # ---- smoking (A already has smoker = ever 100 cigarettes)
    never, former = r.SMQ020 == 2, (r.SMQ020 == 1) & (r.SMQ040 == 3)
    current = (r.SMQ020 == 1) & r.SMQ040.isin([1, 2])
    out["smoking_status"] = _categorical([never, former, current], ["never", "former", "current"])
    cigs = _na(r.SMD650, [777, 999])
    cigs = cigs.where(~(current & (r.SMD641 == 0)), 0.0)        # current, none in past 30 days
    out["cigs_per_day"] = np.where(never | former, 0.0, np.where(current, cigs, np.nan))

    # ---- alcohol
    abstainer = (r.ALQ101 == 2) & (r.ALQ110 == 2)
    freq_q = _na(r.ALQ120Q, [777, 999])
    no_drink_12m = abstainer | (freq_q == 0)
    drinks_day = _na(r.ALQ130, [777, 999]).clip(upper=15)
    per_week = _per_week(freq_q, _na(r.ALQ120U, [7, 9]))
    out["drinks_per_week"] = np.where(no_drink_12m, 0.0, per_week * drinks_day)
    binge = _na(r.ALQ141Q, [777, 999]) * _na(r.ALQ141U, [7, 9]).map({1.0: 52, 2.0: 12, 3.0: 1})
    binge = binge.where(_na(r.ALQ141Q, [777, 999]) != 0, 0.0)
    out["binge_days_per_year"] = np.where(no_drink_12m, 0.0, binge)
    drank = (r.ALQ101 == 1) | (r.ALQ110 == 1)
    out["drinking_status"] = _categorical(
        [abstainer, drank & (freq_q == 0), drank & (freq_q > 0)],
        ["lifetime_abstainer", "former", "current"])

    # ---- diet quality: behaviour questions + energy-adjusted day-1 recall (reliable recalls only)
    out["diet_self_rated"] = _na(r.DBQ700, [7, 9])              # 1 excellent .. 5 poor
    meals_out = _na(r.DBD895, [7777, 9999]).replace(5555, 22)
    out["meals_out_week"] = meals_out
    fast = _na(r.DBD900, [7777, 9999]).replace(5555, 22)
    out["fast_food_week"] = fast.where(meals_out != 0, 0.0)     # gate: no meals out -> 0
    for src, name in [("DBD905", "ready_meals_month"), ("DBD910", "frozen_meals_month")]:
        out[name] = _na(r[src], [7777, 9999]).replace(6666, 90).clip(upper=90)
    ok = (r.DR1DRSTZ == 1) & (r.DR1TKCAL > 0)
    kcal = r.DR1TKCAL.where(ok)
    out["energy_kcal"] = kcal
    out["fiber_g_per_1000kcal"] = r.DR1TFIBE.where(ok) / kcal * 1000
    out["sugar_pct_kcal"] = r.DR1TSUGR.where(ok) * 4 / kcal * 100
    out["satfat_pct_kcal"] = r.DR1TSFAT.where(ok) * 9 / kcal * 100
    out["carb_pct_kcal"] = r.DR1TCARB.where(ok) * 4 / kcal * 100

    # ---- hypertension diagnosis / medication
    out["htn_told"] = _yes_no(r.BPQ020)
    on_med = r.BPQ050A == 1
    not_on = (r.BPQ020 == 2) | (r.BPQ040A == 2) | (r.BPQ050A == 2)
    out["htn_med_now"] = np.where(on_med, 1.0, np.where(not_on, 0.0, np.nan))

    # ---- gestational diabetes history (0 for men and never-pregnant women: no such history)
    no_preg_history = (r.RIAGENDR == 1) | (female & (r.RHQ131 == 2))
    out["gdm_history"] = np.where(no_preg_history, 0.0,
                                  r.RHQ162.map({1.0: 1.0, 2.0: 0.0, 3.0: 0.0}))
    out["baby_9lb"] = np.where(no_preg_history, 0.0, _yes_no(r.RHQ172))

    # ---- weight history (self-reported, kg; differences use self-reported current weight)
    now = _na(r.WHD020, [7777, 9999])
    out["weight_change_1y_kg"] = (now - _na(r.WHD050, [7777, 9999])) * LB_TO_KG
    for src, min_age, name in [("WHD110", 36, "weight_change_10y_kg"),
                               ("WHD120", 27, "weight_gain_since_25_kg")]:
        na = age < min_age                                       # not asked: too young
        out[name] = np.where(na, 0.0, (now - _na(r[src], [7777, 9999])) * LB_TO_KG)
        out[f"{name}_na"] = na.astype(float)                     # 0 here is not "no change"
    out["below_max_weight_kg"] = (_na(r.WHD140, [7777, 9999]) - now) * LB_TO_KG
    out["tried_lose_weight"] = _yes_no(r.WHQ070)

    # ---- set C: education, income, food security
    out["education"] = _na(r.DMDEDUC2, [7, 9])                  # 1 < 9th grade .. 5 college
    out["income_poverty_ratio"] = r.INDFMPIR                    # top-coded at 5
    out["food_insecurity_adult"] = r.FSDAD                       # 1 full .. 4 very low

    # ---- set D: healthcare access / insurance (sensitivity set only)
    out["insured"] = _yes_no(r.HIQ011)
    out["routine_care_place"] = r.HUQ030.map({1.0: 1.0, 3.0: 1.0, 2.0: 0.0})
    visits = _na(_col(r, "HUQ050"), [77, 99]).fillna(
        _na(_col(r, "HUQ051"), [77, 99]).map(HUQ051_TO_HUQ050))
    out["healthcare_visits_bin"] = visits                         # 0,1,2-3,4-9,10-12,13+
    out["hospital_overnight"] = _yes_no(r.HUQ071)
    return out


# Structural rules: rows where the derived value is fixed by a gate answer (or age), so it must
# not be missing when it reaches the imputer.
STRUCTURAL = {
    "pa_met_min_week": lambda r: np.logical_and.reduce([r[g] == 2 for g, *_ in PA_DOMAINS.values()]),
    "cigs_per_day": lambda r: (r.SMQ020 == 2) | ((r.SMQ020 == 1) & (r.SMQ040 == 3)),
    "smoking_status": lambda r: r.SMQ020 == 2,
    "drinks_per_week": lambda r: ((r.ALQ101 == 2) & (r.ALQ110 == 2)) | (r.ALQ120Q == 0),
    "binge_days_per_year": lambda r: ((r.ALQ101 == 2) & (r.ALQ110 == 2)) | (r.ALQ120Q == 0),
    "drinking_status": lambda r: (r.ALQ101 == 2) & (r.ALQ110 == 2),
    "fast_food_week": lambda r: r.DBD895 == 0,
    "htn_med_now": lambda r: r.BPQ020 == 2,
    "gdm_history": lambda r: (r.RIAGENDR == 1) | (r.RHQ131 == 2),
    "baby_9lb": lambda r: (r.RIAGENDR == 1) | (r.RHQ131 == 2),
    "weight_change_10y_kg": lambda r: r.RIDAGEYR < 36,
    "weight_gain_since_25_kg": lambda r: r.RIDAGEYR < 27,
}


def assert_no_structural_missing(raw: pd.DataFrame, derived: pd.DataFrame) -> None:
    """Raise if a skip-pattern feature is missing where its gate fixes the value."""
    bad = {}
    for col, rule in STRUCTURAL.items():
        mask = np.asarray(rule(raw), dtype=bool)
        n = int(derived.loc[mask, col].isna().sum())
        if n:
            bad[col] = n
    if bad:
        raise ValueError(f"structural missingness would reach the imputer: {bad}")
