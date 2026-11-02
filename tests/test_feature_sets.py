"""Feature sets: nested, no lab/label/gate columns, exclusions respected, D only via the rule."""
import pandas as pd
import pytest
import yaml

from src.evaluation.selection import base_for_d, primary_model, primary_set, selection_record
from src.evaluation.temporal import feature_set

CFG = yaml.safe_load(open("configs/config.yaml"))
EXCLUDED_ITEMS = {"DIQ160", "DIQ170", "DIQ172", "DIQ180", "MCQ365A", "MCQ365B", "MCQ365C",
                  "MCQ365D", "MCQ370A", "MCQ370B", "MCQ370C", "MCQ370D", "RIDEXPRG"}
ACCESS = {"insured", "routine_care_place", "healthcare_visits_bin", "hospital_overnight"}


def test_sets_are_nested():
    a, b, c = (set(feature_set(CFG, s)[0]) for s in "ABC")
    assert a < b < c
    assert a == set(CFG["features"]["numeric"] + CFG["features"]["categorical"])


def test_excluded_items_and_access_only_in_d():
    for s in "ABC":
        feats = set(feature_set(CFG, s)[0])
        assert not feats & EXCLUDED_ITEMS and not feats & ACCESS
        assert not any("pregnan" in f for f in feats)
    for base in "BC":
        d = set(feature_set(CFG, "D", base)[0])
        assert d == set(feature_set(CFG, base)[0]) | ACCESS


def test_d_requires_resolved_base():
    with pytest.raises(ValueError, match="base_for_d"):
        feature_set(CFG, "D")


def test_guard_rejects_gate_and_lab_columns():
    bad = {**CFG, "feature_sets": {**CFG["feature_sets"],
                                   "A": {"numeric": ["age", "gate_SMQ020"], "categorical": []}}}
    with pytest.raises(ValueError, match="gate"):
        feature_set(bad, "A")
    bad["feature_sets"]["A"] = {"numeric": ["age", "LBXGLU"], "categorical": []}
    with pytest.raises(ValueError, match="lab-derived"):
        feature_set(bad, "A")


def summary(rows):
    return pd.DataFrame(rows, columns=["feature_set", "model", "strategy", "pr_auc_mean",
                                       "pr_auc_sd"])


def test_selection_rules():
    s = summary([
        ("A", "logreg", "none", 0.090, 0.010), ("A", "xgboost", "none", 0.088, 0.010),
        ("B", "logreg", "none", 0.096, 0.008), ("B", "xgboost", "none", 0.104, 0.009),
        ("C", "logreg", "none", 0.097, 0.008), ("C", "xgboost", "none", 0.105, 0.012),
        ("A", "xgboost", "smotenc", 0.200, 0.001),          # not a main combination: ignored
    ])
    ps = primary_set(s, ["logreg", "xgboost"])
    # best = C/xgboost 0.105 - 0.012 = 0.093: A (0.090) misses, B (0.104) is within -> B
    assert ps["primary_set"] == "B" and ps["best_set"] == "C"
    pm = primary_model(s, "B", ["logreg", "xgboost"])
    # on B: best xgboost 0.104 - 0.009 = 0.095; logreg 0.096 is within -> simplest = logreg
    assert pm["primary_model"] == "logreg" and pm["best_model"] == "xgboost"
    assert base_for_d(s, ["logreg", "xgboost"]) == "C"     # mean C 0.101 > B 0.100
    # the record keeps each rule's numbers apart (both rules use best_model, threshold, ...)
    rec = selection_record(ps, pm, "C")
    assert rec["primary_set"] == "B" and rec["primary_model"] == "logreg" and rec["d_base"] == "C"
    assert rec["set_rule"]["best_set"] == "C" and rec["set_rule"]["best_model"] == "xgboost"
    assert rec["set_rule"]["best_pr_auc"] == 0.105 and abs(rec["set_rule"]["threshold"] - 0.093) < 1e-12
    assert rec["model_rule"]["feature_set"] == "B" and rec["model_rule"]["best_pr_auc"] == 0.104
    assert abs(rec["model_rule"]["threshold"] - 0.095) < 1e-12
