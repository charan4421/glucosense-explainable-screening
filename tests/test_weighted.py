"""Survey-weighted metrics: weights change the estimate, constant weights do not."""
import numpy as np
import pandas as pd

from scripts.dev_weighted import weighted_table
from src.data.nhanes import MAIN_LABEL
from src.evaluation.metrics import bootstrap_ci, summarize


def _data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    p = np.clip(0.03 + 0.1 * y + rng.normal(0, 0.03, n), 0.001, 0.999)
    return y, p, rng


def test_constant_weights_equal_unweighted():
    y, p, _ = _data()
    u, w = summarize(y, p), summarize(y, p, sample_weight=np.full(len(y), 3.7))
    assert all(np.isclose(u[k], w[k]) for k in u)


def test_weights_matter():
    y, p, rng = _data()
    w = np.where(y == 1, 0.2, 1.0) * rng.uniform(0.5, 2, len(y))
    s = summarize(y, p, sample_weight=w)
    assert s["prevalence"] < y.mean() and not np.isclose(s["pr_auc"], summarize(y, p)["pr_auc"])
    ci = bootstrap_ci(y, p, 100, sample_weight=w)
    assert ci["pr_auc_ci_low"] <= s["pr_auc"] <= ci["pr_auc_ci_high"]


def test_pooled_divisor_and_rows():
    y, p, rng = _data(n=900)
    dev = pd.DataFrame({MAIN_LABEL: y, "cycle": np.repeat(["c1", "c2", "c3"], 300),
                        "wtmec2yr": rng.uniform(5e3, 5e4, 900)})
    t = weighted_table(dev, p, n_boot=20, seed=1)
    assert len(t) == 8 and set(t.scope) == {"pooled", "c1", "c2", "c3"}
    pooled = t[(t.scope == "pooled") & (t.weighting == "WTMEC2YR")].iloc[0]
    w = dev.wtmec2yr / 3
    assert np.isclose(pooled.prevalence, (w * y).sum() / w.sum())
