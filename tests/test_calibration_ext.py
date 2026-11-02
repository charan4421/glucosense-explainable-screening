"""Race/ethnicity calibration extension: one-hot terms, group offsets, decision rule."""
import numpy as np
import pandas as pd

from src.evaluation.calibration_ext import (
    RaceTerms,
    apply_offsets,
    decide,
    group_offsets,
)


def test_race_terms_replace_ordinal_code():
    X = pd.DataFrame({"race_eth": ["1", "3", "6"], "bmi": [30.0, 25.0, 22.0],
                      "waist_cm": [100.0, 90.0, 80.0]})
    Z = RaceTerms(interactions=True).fit_transform(X)
    assert "race_eth" not in Z and "race_3" not in Z            # reference has no dummy
    assert list(Z.race_1) == [1, 0, 0] and list(Z.race_6) == [0, 0, 1]
    assert list(Z.race_6_x_bmi) == [0, 0, 22.0] and list(Z.race_1_x_waist) == [100.0, 0, 0]
    assert "race_1_x_bmi" not in RaceTerms(interactions=False).fit_transform(X)


def test_offsets_restore_calibration_in_large_groups_only():
    rng = np.random.default_rng(0)
    g = np.repeat(["big_over", "big_under", "small"], [3000, 3000, 200])
    true = np.where(g == "big_over", 0.02, np.where(g == "big_under", 0.08, 0.05))
    y = (rng.random(len(g)) < true).astype(int)
    p = np.full(len(g), 0.05)                                 # same prediction for everyone
    off = group_offsets(g, y, p, min_pos=30)
    assert off["big_over"] < 0 < off["big_under"] and off["small"] == 0.0
    q = apply_offsets(g, p, off)
    for level in ["big_over", "big_under"]:
        m = g == level
        assert np.isclose(q[m].mean(), y[m].mean(), rtol=1e-6)   # O/E = 1 after the offset


def _sub(cis):
    return pd.DataFrame([{"variable": "race_eth", "group": k, "observed_expected_ci_low": lo,
                          "observed_expected_ci_high": hi} for k, (lo, hi) in cis.items()])


def test_decision_rule():
    groups = ["Mexican American", "Other Hispanic", "Non-Hispanic White", "Non-Hispanic Black",
              "Non-Hispanic Asian"]
    primary = _sub(dict(zip(groups, [(1.1, 1.6), (0.9, 1.6), (0.5, 0.7), (0.9, 1.3), (1.3, 2.0)], strict=True)))
    better = _sub(dict(zip(groups, [(0.9, 1.3)] * 5, strict=True)))
    same = primary.copy()
    out = decide(primary, (0.0938, 0.0153), {"better_but_weak": (better, 0.070),
                                             "better": (better, 0.090), "same": (same, 0.095)})
    assert out["primary_n_covered"] == 2
    assert not out["candidates"]["better_but_weak"]["qualifies"]       # PR-AUC below mean - SD
    assert not out["candidates"]["same"]["qualifies"]                  # no more groups covered
    assert out["adopted"] == "better"
    assert decide(primary, (0.0938, 0.0153), {"same": (same, 0.095)})["result"] == "negative result"
