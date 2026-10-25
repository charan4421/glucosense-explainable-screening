"""Label rules for the NHANES analysis sample (no downloads needed)."""
import numpy as np
import pandas as pd

from src.data.nhanes import MAIN_LABEL, SENS_LABEL, label_frame


def raw(**over):
    row = dict(SEQN=0, RIDAGEYR=45, DIQ010=2, DIQ050=2, DIQ070=np.nan,
               LBXGH=5.5, LBXGLU=np.nan, WTSAF2YR=np.nan)
    row.update(over)
    return row


def test_sample_rules_and_labels():
    rows = [
        raw(SEQN=1),                                    # kept, negative
        raw(SEQN=2, LBXGH=6.5),                         # kept, positive at exactly 6.5
        raw(SEQN=3, LBXGH=6.4, LBXGLU=130, WTSAF2YR=5e4),  # main negative, sensitivity positive
        raw(SEQN=4, DIQ010=3, LBXGH=7.0),               # borderline kept
        raw(SEQN=5, DIQ010=1, LBXGH=8.0),               # diagnosed -> excluded
        raw(SEQN=6, DIQ010=7),                          # refused -> excluded
        raw(SEQN=7, DIQ010=9),                          # don't know -> excluded
        raw(SEQN=8, DIQ050=1, LBXGH=7.2),               # insulin, no diagnosis -> excluded
        raw(SEQN=9, DIQ010=3, DIQ070=1, LBXGH=6.8),     # pills -> excluded
        raw(SEQN=10, LBXGH=np.nan, LBXGLU=140, WTSAF2YR=5e4),  # no HbA1c -> excluded
        raw(SEQN=11, RIDAGEYR=19, LBXGH=7.0),           # not adult -> excluded
        raw(SEQN=12, LBXGLU=150, WTSAF2YR=0),           # FPG but zero fasting weight -> not fasting
    ]
    out = label_frame(pd.DataFrame(rows)).set_index("SEQN")

    assert sorted(out.index) == [1, 2, 3, 4, 12]
    assert out[MAIN_LABEL].to_dict() == {1: 0, 2: 1, 3: 0, 4: 1, 12: 0}
    assert out.loc[3, SENS_LABEL] == 1
    assert out[SENS_LABEL].isna().sum() == 4            # only SEQN 3 is in the fasting subsample
