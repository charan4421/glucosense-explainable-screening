"""Subgroup table: small groups are flagged, metrics need both classes."""
import numpy as np
import pandas as pd

from src.evaluation.subgroups import subgroup_table


def test_flags_groups_with_few_positives():
    rng = np.random.default_rng(0)
    n = 4000
    g = pd.Series(np.where(np.arange(n) < 3800, "big", "small"))
    y = np.zeros(n, dtype=int)
    y[:100] = 1                       # 100 positives in "big"
    y[3800:3810] = 1                  # 10 positives in "small"
    p = np.clip(0.03 + 0.2 * y + rng.normal(0, 0.02, n), 0.001, 0.999)
    t = subgroup_table(y, p, {"g": g}, min_pos=30).set_index("group")
    assert t.loc["big", "reliable"] and t.loc["big", "n_pos"] == 100
    assert not t.loc["small", "reliable"] and "not reliable" in t.loc["small", "note"]
    assert t.loc["all", "n"] == n and t.loc["all", "n_pos"] == 110


def test_group_with_one_class_has_no_metrics():
    y = np.array([0, 0, 0, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.1, 0.8, 0.3, 0.7])
    t = subgroup_table(y, p, {"g": pd.Series(list("aaabbb"))}, min_pos=1).set_index("group")
    assert pd.isna(t.loc["a", "pr_auc"]) and not pd.isna(t.loc["b", "pr_auc"])
