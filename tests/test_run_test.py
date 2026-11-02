"""Test-stage helpers (synthetic data only) and the 2017-2018 'left missing' alcohol mapping."""
import numpy as np
import pandas as pd

from scripts.run_test import metric_table, recalibrate, weighted_metrics
from src.data.derive import assert_no_structural_missing, derive
from src.data.nhanes import UNMAPPED
from tests.test_derive import frame, raw_row


def synthetic(n=3000, seed=0, shift=-0.7):
    rng = np.random.default_rng(seed)
    lp = rng.normal(-3.5, 1.0, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-lp))).astype(int)
    p = 1 / (1 + np.exp(-(lp - shift)))                 # risks overpredicted by exp(-shift)
    return y, p


def test_recalibration_restores_calibration_in_the_large():
    y, p = synthetic()
    q, a = recalibrate(y, p)
    assert a < 0                                         # overprediction -> negative intercept
    assert abs(q.mean() - y.mean()) < 1e-6
    assert np.array_equal(np.argsort(q), np.argsort(p))  # ranking unchanged


def test_metric_table_intervals_bracket_point_and_ranking_unchanged_by_recal():
    y, p = synthetic()
    raw, rec = metric_table(y, p, n_boot=50), metric_table(y, p, n_boot=50, recal=True)
    for k in ["pr_auc", "roc_auc", "brier", "observed_expected", "calibration_slope"]:
        assert raw[f"{k}_ci_low"] <= raw[k] <= raw[f"{k}_ci_high"], k
    assert raw["pr_auc"] == rec["pr_auc"] and raw["roc_auc"] == rec["roc_auc"]
    assert rec["brier"] < raw["brier"]
    assert abs(rec["observed_expected"] - 1) < 1e-6


def test_weighted_metrics_unit_weights_match_unweighted():
    y, p = synthetic()
    w = weighted_metrics(y, p, np.ones(len(y)), n_boot=20)
    u = metric_table(y, p, n_boot=20)
    assert np.isclose(w["roc_auc"], u["roc_auc"]) and np.isclose(w["pr_auc"], u["pr_auc"])
    assert np.isclose(w["weighted_prevalence"], y.mean())


def test_unmapped_alcohol_is_left_missing_not_zero():
    cols = UNMAPPED["2017-2018"]
    raw = frame(raw_row(**{c: np.nan for c in cols}, ALQ130=3))
    d = derive(raw)
    for c in ["drinks_per_week", "binge_days_per_year", "drinking_status"]:
        assert pd.isna(d.loc[0, c]), c
    assert d.loc[0, "pa_met_min_week"] == 0                # other items unaffected
    assert_no_structural_missing(raw, d)                   # missing is not structural here
