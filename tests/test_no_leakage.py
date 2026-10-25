"""Leakage guards: lab-derived columns never become features; the temporal split looks only back."""
import pandas as pd
import pytest
import yaml

from src.data.nhanes import MAIN_LABEL, SENS_LABEL
from src.evaluation.temporal import LAB_DERIVED, feature_columns, temporal_split


def load_cfg():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def test_config_features_contain_no_lab_derived_columns():
    feats = feature_columns(load_cfg())
    assert {"LBXGH", "LBXGLU", MAIN_LABEL, SENS_LABEL} <= LAB_DERIVED
    bad = [f for f in feats if f in LAB_DERIVED or f.upper() in LAB_DERIVED]
    assert not bad, f"lab-derived columns in the feature list: {bad}"


@pytest.mark.parametrize("col", ["LBXGH", "lbxglu", MAIN_LABEL, SENS_LABEL])
def test_guard_rejects_lab_derived_feature(col):
    cfg = load_cfg()
    cfg["features"]["numeric"] = cfg["features"]["numeric"] + [col]
    with pytest.raises(ValueError, match="lab-derived"):
        feature_columns(cfg)


def test_temporal_split_trains_only_on_earlier_cycles():
    df = pd.DataFrame({"cycle": ["2011-2012", "2015-2016", "2017-2018"], "x": [1, 2, 3]})
    train, test = temporal_split(df, "2017-2018")
    assert set(train.cycle) == {"2011-2012", "2015-2016"} and set(test.cycle) == {"2017-2018"}
    with pytest.raises(ValueError, match="later"):
        temporal_split(df, "2015-2016")
