# GlucoSense-AI

Calibrated, explainable screening for **undiagnosed diabetes** from non-laboratory features.

- **Main data:** NHANES 2011-2018 (4 cycles). Lab values and survey weights are used only to build labels and prevalence estimates, never as features.
  - **Analysis sample:** adults >= 20 y with HbA1c measured and no self-reported diagnosis (DIQ010 not 1). Excluded: refused/don't know on DIQ010 (7, 9) and anyone taking insulin (DIQ050) or diabetes pills (DIQ070) without a diagnosis. Borderline (DIQ010 = 3) is kept.
  - **Main label** `undiagnosed_diabetes`: HbA1c >= 6.5 %.
  - **Sensitivity label** `undiagnosed_diabetes_a1c_or_fpg` (sensitivity analysis only): HbA1c >= 6.5 % or fasting plasma glucose >= 126 mg/dL, defined only for the fasting subsample (missing otherwise).
  - **Prevalence** is reported unweighted and survey-weighted, per cycle and pooled: WTMEC2YR for the main label, WTSAF2YR for the sensitivity label, each divided by the number of cycles when pooling. Current build (v2, development cycles 2011-2016 only): main label 3.1 % unweighted / 2.1 % weighted (pooled, n = 13,189, 410 positives); sensitivity label 5.6 % / 4.0 % (fasting subsample, n = 5,905).
- **External validation:** BRFSS (overlapping features only).
- **Validation design:** nested 5-fold CV, plus temporal split (train on earlier NHANES cycles, test on the latest).
- **Models:** XGBoost, PyTorch MLP, TabNet.
- **Baselines** (same nested-CV folds and preprocessing): L2 logistic regression with tuned C, and an **ADA-style** risk score. The ADA-style score approximates the ADA Type 2 Diabetes Risk Test with the columns available; it is not the official score.
  - Included: age, sex, family history (MCQ300C, any close relative), BMI (official cut-offs), and high blood pressure as a **proxy** (measured SBP >= 140 or DBP >= 90; the official item is a diagnosis or medication).
  - Missing (scored 0): physical activity and gestational diabetes. Maximum 9 of the official 11 points; a missing value scores 0.
  - Points are mapped to risk by a one-variable logistic regression fit on each training fold.
- **Imbalance strategies compared:** none, class weighting, SMOTE-NC (always applied inside CV folds).
- **Metrics:** PR-AUC (primary), ROC-AUC, Brier score, calibration curve, Decision Curve Analysis.
- **Explainability:** SHAP (global + local), LIME consistency check, DiCE counterfactuals with immutable features.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.data.nhanes --download                 # builds 2011-2016 only (dev data v2)
# development analysis v2 (docs/analysis_plan.md): models x feature sets A-D, 5-fold x 3
# repeats nested CV, leave-one-cycle-out; checkpointed to results/dev_cv_v2/
python scripts/run_dev.py
# v1 (earlier, set A only): nested CV on 2011-2016 only; each model/strategy is checkpointed to results/dev_cv/ and
# skipped on restart if config + seed + n_iter are unchanged
python scripts/run_experiment.py --config configs/config.yaml --n-iter 5
python scripts/run_experiment.py --models logreg,ada_style --strategies none --n-iter 5   # baselines
python scripts/summarize_dev_cv.py                   # rebuild results/dev_cv_summary.csv any time
# temporal test: tune on 2011-2016, score once on 2017-2018 (reports/temporal_test.csv)
python scripts/run_experiment.py --mode temporal --models xgboost --strategies none
```

Results (metrics CSV, plots) are written to `reports/`.

## Layout

```
configs/      experiment settings
src/data/     NHANES + BRFSS loaders
src/preprocessing/  KNN imputation, RobustScaler, SMOTE-NC pipeline
src/models/   XGBoost, MLP, TabNet wrappers (sklearn-compatible)
src/evaluation/     nested CV, metrics, DCA
src/explain/  SHAP, LIME, DiCE
src/app/      FastAPI service + Streamlit dashboard
tests/        synthetic-data tests (no downloads needed)
```

## Notes on honesty

- Counterfactuals are model explanations, not causal advice.
- SMOTE-NC distorts probabilities; the calibration cost is reported, not hidden.
- Nested CV reduces optimistic bias; it does not remove it.
- SDs across repeated CV folds are not independent: training sets overlap, so they understate
  the uncertainty. Reported SDs describe spread, not standard errors. Model and set differences
  are compared on identical folds with the Nadeau-Bengio corrected 95 % interval instead.

## Limitations (draft)

- **Survey design not modelled.** Models are trained and scored unweighted on the NHANES
  sample. Survey weights (WTMEC2YR) enter only prevalence and a weighted sensitivity analysis of
  ROC-AUC and PR-AUC; the strata and PSUs (SDMVSTRA, SDMVPSU) are not used at all.
- **Intervals are too narrow.** Bootstrap intervals resample persons independently, ignoring
  strata and PSUs, so they understate the uncertainty. SDs across repeated CV folds are not
  independent either (see above).
- **Sample vs population prevalence.** In the 2011-2016 development data the main label has a
  prevalence of 3.1 % unweighted but 2.1 % weighted. Unweighted PR-AUC, PPV and net benefit
  describe the NHANES sample, not the US adult population; read weighted PR-AUC against 2.1 %.
- **Calibration by race/ethnicity: the gap came from an encoding error.** Logistic regression
  first received race/ethnicity (and, in sets B-D, smoking and drinking status) as one ordinal
  code, so it fitted a single slope across nominal groups. The development subgroup analysis
  exposed it; it was fixed by one-hot encoding before the test (docs/analysis_plan.md,
  Deviations 2026-11-02), and the selection rules, applied again, still pick logistic
  regression on set A. Out-of-fold O/E for that model (95 % bootstrap CI, 1000 resamples within
  group; old results in `results/archive_v2_ordinal_logreg/`):

  | Group (positives) | O/E, ordinal (before) | O/E, one-hot (after) |
  |---|---|---|
  | Non-Hispanic White (94) | 0.59 (0.48-0.71) | 0.97 (0.79-1.17) |
  | Non-Hispanic Asian (65) | 1.66 (1.31-2.08) | 1.04 (0.82-1.30) |
  | Non-Hispanic Black (116) | 1.15 (0.96-1.37) | 1.00 (0.83-1.19) |
  | Mexican American (72) | 1.37 (1.06-1.68) | 1.01 (0.78-1.24) |
  | Other Hispanic (53) | 1.28 (0.96-1.64) | 1.02 (0.77-1.30) |
  | CI includes 1 | 2 of 5 groups | 5 of 5 groups |
  | Nested-CV PR-AUC / ROC-AUC (all) | 0.094 / 0.783 | 0.104 / 0.805 |

  "Other / multiracial" (10 positives) is too small to judge. The intervals are wide (about 20-25 %
  either side) and ignore the survey design, so this shows the gross
  miscalibration is gone, not that calibration is equal across groups. Group-wise
  recalibration was dropped as no longer needed. The MLP also received ordinal codes and was
  not refit; it is never the primary model.
- **Label based on HbA1c only.** The main label is HbA1c ≥ 6.5 % from a single measurement, with
  no confirmatory test. HbA1c misses some cases that fasting glucose or an OGTT would find, and
  differs by race/ethnicity and with anaemia or haemoglobin variants. HbA1c-or-FPG on the
  fasting subsample is a sensitivity analysis only.
