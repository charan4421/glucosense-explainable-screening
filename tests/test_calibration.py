"""Calibration statistics and bootstrap CIs on synthetic data with a known answer."""
import numpy as np
import pytest

from src.evaluation.metrics import bootstrap_ci, calibration_stats, plot_calibration, summarize


def calibrated(n=40_000, seed=0):
    rng = np.random.default_rng(seed)
    p = 1 / (1 + np.exp(-rng.normal(-3, 1, n)))       # ~5-6 % base rate, like the real data
    y = (rng.random(n) < p).astype(int)               # outcomes drawn from p: calibrated by design
    return y, p


def test_calibrated_predictions_give_ideal_stats():
    y, p = calibrated()
    s = calibration_stats(y, p)
    assert s["slope"] == pytest.approx(1, abs=0.06)
    assert s["intercept"] == pytest.approx(0, abs=0.2)
    assert s["intercept_offset"] == pytest.approx(0, abs=0.05)
    assert s["citl_difference"] == pytest.approx(0, abs=0.005)


def test_miscalibration_moves_stats_the_right_way():
    y, p = calibrated()
    lp = np.log(p / (1 - p))
    under = calibration_stats(y, 1 / (1 + np.exp(-(lp - 0.5))))     # risks too low
    assert under["intercept_offset"] > 0.3 and under["citl_ratio_observed_expected"] > 1
    extreme = calibration_stats(y, 1 / (1 + np.exp(-2 * (lp + 3) + 3)))   # too extreme
    assert extreme["slope"] < 0.7


def test_bootstrap_ci_is_reproducible_and_brackets_the_estimate():
    y, p = calibrated(n=4000)
    a, b = bootstrap_ci(y, p, n_boot=200, seed=1), bootstrap_ci(y, p, n_boot=200, seed=1)
    assert a == b
    point = summarize(y, p)
    for k in ["pr_auc", "roc_auc", "brier"]:
        assert a[f"{k}_ci_low"] < point[k] < a[f"{k}_ci_high"]


def test_plot_calibration_writes_file(tmp_path):
    y, p = calibrated(n=3000)
    out = tmp_path / "cal.png"
    plot_calibration(y, p, out)
    assert out.exists() and out.stat().st_size > 10_000
