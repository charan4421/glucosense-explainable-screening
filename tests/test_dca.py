"""Decision curve table: window, reference strategies and net-benefit arithmetic."""
import numpy as np

from scripts.dev_dca import THRESHOLDS, dca_table


def test_window_is_one_to_five_percent():
    assert THRESHOLDS[0] == 0.01 and THRESHOLDS[-1] == 0.05


def test_net_benefit_arithmetic():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.02, 0.5, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005])
    t = dca_table(y, {"m": p}, thresholds=np.array([0.01, 0.03]))
    # t=0.01: treated = rows 0,1,2 -> TP 2, FP 1 -> 2/10 - 1/10 * 0.01/0.99
    assert np.isclose(t.m[0], 0.2 - 0.1 * 0.01 / 0.99)
    # t=0.03: treated = rows 0,2 -> TP 1, FP 1
    assert np.isclose(t.m[1], 0.1 - 0.1 * 0.03 / 0.97)
    assert np.allclose(t.treat_all, 0.2 - 0.8 * t.threshold / (1 - t.threshold))
    assert (t.treat_none == 0).all()


def test_bootstrap_bands_cover_estimate_and_are_paired():
    rng = np.random.default_rng(1)
    y = (rng.random(3000) < 0.05).astype(int)
    p = np.clip(0.03 + 0.1 * y + rng.normal(0, 0.02, len(y)), 0.001, 0.999)
    t = dca_table(y, {"m": p}, n_boot=200)
    for c in ["m", "treat_all", "m_minus_treat_all"]:
        assert (t[f"{c}_ci_low"] <= t[c] + 1e-12).all() and (t[c] <= t[f"{c}_ci_high"] + 1e-12).all()
    # paired: the difference band is narrower than the two separate bands combined
    assert ((t.m_minus_treat_all_ci_high - t.m_minus_treat_all_ci_low)
            < (t.m_ci_high - t.m_ci_low) + (t.treat_all_ci_high - t.treat_all_ci_low)).all()


def test_bootstrap_is_reproducible():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 0] * 30)
    p = np.linspace(0.001, 0.2, len(y))
    a, b = dca_table(y, {"m": p}, n_boot=50, seed=7), dca_table(y, {"m": p}, n_boot=50, seed=7)
    assert a.equals(b)
