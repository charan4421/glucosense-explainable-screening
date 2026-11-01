"""ADA-style risk score: an approximation of the ADA Type 2 Diabetes Risk Test (Bang et al.,
Ann Intern Med 2009) built only from columns this dataset has. It is NOT the official score.

Items and points (official test: 0-11, >= 5 = high risk):

    item                          official points                 here
    age                           <40: 0, 40-49: 1, 50-59: 2,     included as official
                                  >=60: 3
    sex                           male: 1                         included as official
    family history of diabetes    parent or sibling: 1            MCQ300C (close relative; NHANES
                                                                  wording is broader than parent/sibling)
    high blood pressure           diagnosed or on medication: 1   PROXY: measured mean SBP >= 140 or
                                                                  DBP >= 90 (no diagnosis/medication
                                                                  item loaded)
    BMI (from weight/height chart) 25-<30: 1, 30-<40: 2, >=40: 3  measured BMI, same cut-offs
    physically active             no: 1                           MISSING (PAQ not loaded): 0 points
    gestational diabetes (women)  yes: 1                          MISSING (RHQ not loaded): 0 points

Our maximum is 9 of 11. A missing item value scores 0 points. Points are mapped to a predicted
risk by a one-variable logistic regression fit on the training data only, so calibration, Brier
and O/E are comparable with the ML models; ranking metrics depend on the points alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

ADA_ITEMS_INCLUDED = ["age", "sex", "family_history_diabetes", "hypertension (BP proxy)", "bmi"]
ADA_ITEMS_MISSING = ["physical_activity", "gestational_diabetes"]


def ada_style_points(X: pd.DataFrame) -> np.ndarray:
    age, bmi = X["age"], X["bmi"]
    pts = (np.select([age >= 60, age >= 50, age >= 40], [3, 2, 1], 0)
           + (X["sex"] == "male").astype(int)
           + (X["family_history_diabetes"] == "yes").astype(int)
           + ((X["systolic_bp"] >= 140) | (X["diastolic_bp"] >= 90)).astype(int)
           + np.select([bmi >= 40, bmi >= 30, bmi >= 25], [3, 2, 1], 0))   # NaN compares False -> 0
    return np.asarray(pts, dtype=float)


# Mixin before BaseEstimator, as for the other wrappers (sklearn >= 1.6 tag order).
class ADAStyleScore(ClassifierMixin, BaseEstimator):
    """Fixed points; only the points -> risk mapping is learned. Takes the raw feature frame."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.mapper_ = LogisticRegression(C=np.inf, max_iter=1000).fit(
            ada_style_points(X).reshape(-1, 1), np.asarray(y))
        return self

    def predict_proba(self, X):
        return self.mapper_.predict_proba(ada_style_points(X).reshape(-1, 1))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
