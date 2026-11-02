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
# the single 2017-2018 temporal test (docs/analysis_plan.md section 9; already run, do not rerun)
python -m src.data.nhanes --test                     # builds 2017-2018 alone, same rules
python scripts/run_test.py --stage fit               # 2011-2016 only: refit, hashes, run log
python scripts/run_test.py --stage score             # refuses unless tree clean, tag pre-test, hashes match
# v1 (before the plan) temporal run of XGBoost, kept as a record (results/temporal_test.csv)
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

## Results: development (2011-2016) and the single 2017-2018 temporal test

> **2017-2018 was not untouched.** Before this analysis plan existed, the old v1 pipeline scored
> an XGBoost model on 2017-2018 once (commit `df1b305`: PR-AUC 0.101, ROC-AUC 0.776, prevalence
> 3.8 %). No v2 selection rule used that result. The test below is the first and only scoring of
> the pre-specified v2 model, but not the first look at the cycle (deviation 2026-11-02 in
> `docs/analysis_plan.md`).

Pre-specified in `docs/analysis_plan.md` (section 9 and the deviations). The test was scored
**once** from the commit tagged `pre-test` (`6795066`); its results are in
`results/test_2017_2018/` (commit `befb8b1`), with every hash in `run_log.json`. The primary
model is L2 logistic regression on set A (10 inputs, one-hot categoricals), refit on all of
2011-2016 (C = 0.3) and **not recalibrated**.

**Primary model, unweighted (primary result).** Test values have bootstrap 95 % intervals (1000
resamples, seed 42). Development values are means over the 15 nested-CV folds; LOCO is the
2015-2016 held-out cycle, the closest development analogue.

| Metric | Dev nested CV | Dev LOCO 2015-16 | **Test 2017-18** |
|---|---|---|---|
| n (positives), prevalence | 13,189 (410), 3.1 % | 4,294 (127), 3.0 % | **4,095 (157), 3.8 %** |
| PR-AUC | 0.104 | 0.079 | **0.107 (0.085-0.139)** |
| PR-AUC / prevalence | 3.3 | 2.7 | **2.8** |
| ROC-AUC | 0.805 | 0.773 | **0.780 (0.749-0.811)** |
| Brier | 0.029 | 0.029 | **0.036 (0.031-0.041)** |
| Calibration slope | 1.00 | 0.83 | **0.89 (0.76-1.02)** |
| Calibration intercept | 0.01 | −0.73 | **−0.33 (−0.74 to 0.05)** |
| O/E | 1.00 | 0.79 | **0.98 (0.84-1.13)** |

Discrimination dropped from development (ROC-AUC 0.805 → 0.780), to about the level the
2015-2016 LOCO predicted, as section 9 expected. Calibration-in-the-large held (O/E 0.98). The
slope of 0.89 points to slightly too-extreme risks, but its interval includes 1. PR-AUC is higher
than in development only because prevalence is higher; relative to prevalence it fell (3.3 → 2.8).

**Secondary results (labelled; none replaces the primary result).**

| Model / version | Dev PR-AUC / ROC-AUC | Test PR-AUC | Test ROC-AUC | Test O/E |
|---|---|---|---|---|
| Primary, survey-weighted (WTMEC2YR) | 0.088 / 0.844 (prev 2.1 %) | 0.082 (0.063-0.114), prev 2.5 % | 0.799 (0.751-0.846) | — |
| Primary, intercept-recalibrated on the test | — | 0.107 (unchanged) | 0.780 (unchanged) | 1.00 by construction |
| EBM / set B | 0.112 / 0.800 | 0.114 (0.090-0.151) | 0.781 (0.747-0.812) | 1.10 (0.94-1.27) |
| ADA-style score (comparator) | 0.079 / 0.749 | 0.072 (0.058-0.087) | 0.717 (0.680-0.752) | 1.03 (0.87-1.19) |

- **Recalibration:** the recalibration intercept was −0.02. It changed Brier only in the fourth
  decimal (0.0360 → 0.0359) and net benefit by less than 0.001, so a local intercept update is not
  needed.
- **EBM / B:** it adds 30 questionnaire inputs, but on the test it is no better than the primary
  model. Its alcohol items were left missing and imputed, because the 2017 questionnaire changed.
- **ADA-style score:** both models beat it on every discrimination metric.

**Decision curve (1-5 %, `test_dca.csv/.png`).** The primary model's net benefit is above
treat-all at every threshold from 1 % to 5 %, and the paired bootstrap band of the difference
excludes 0 throughout.

| Threshold | Net benefit − treat-all, test (95 % band) | Dev (out-of-fold) |
|---|---|---|
| 1 % | 0.0027 (0.0026-0.0029) | 0.0026 |
| 2 % | 0.0050 (0.0029-0.0070) | 0.0080 |
| 3 % | 0.0107 (0.0080-0.0134) | 0.0141 |
| 5 % | 0.0249 (0.0212-0.0287) | 0.0286 |

**Subgroups, primary model (`test_subgroups_logreg__none__A.csv`).**

| Group (positives) | PR-AUC | ROC-AUC | Slope | O/E |
|---|---|---|---|---|
| Women (90) | 0.151 (0.112-0.218) | 0.819 (0.782-0.853) | 1.04 | 1.14 (0.91-1.37) |
| Men (67) | 0.075 (0.054-0.107) | 0.730 (0.675-0.789) | 0.70 (0.53-0.93) | 0.83 (0.64-1.00) |
| Non-Hispanic Black (54) | 0.137 | 0.743 (0.678-0.806) | 0.73 | 1.07 (0.81-1.33) |
| Other groups (8-28 each) | flagged unreliable (< 30 positives) | | | |

**Post-test findings** (seen only after scoring; reported, nothing was rerun or changed):
1. **Men vs women.** Discrimination and calibration slope are clearly worse in men on the test
   (ROC-AUC 0.73 vs 0.82, slope 0.70). In development the two sexes were alike (ROC-AUC 0.805
   vs 0.803, slopes 1.04 and 0.97). This was not expected, and it rests on 67 male positives.
2. **Race/ethnicity cannot be checked on the test.** Only non-Hispanic Black adults reach 30
   positives. So the development fix for calibration by race/ethnicity is not confirmed or
   refuted on 2017-2018.
3. **The recalibrated O/E interval is degenerate (1.00-1.00).** The intercept is re-estimated in
   every resample, which forces O/E to 1. This is a property of the method, not a result.
4. **PR-AUC is not comparable across prevalences.** Read it against prevalence; the ratio fell.

**Post-test, exploratory, not used for any decision** (`scripts/post_test_exploratory.py`,
`results/post_test_exploratory/`). These were computed from the saved test predictions only, with
no model fitted, refitted or rescored. Bootstrap: 1000 resamples, seed 42.

| Comparison | Difference | 95 % CI |
|---|---|---|
| Primary − ADA-style, ROC-AUC (unweighted) | 0.063 | 0.040 to 0.087 |
| Primary − ADA-style, PR-AUC (unweighted) | 0.035 | 0.023 to 0.057 |
| Primary − ADA-style, ROC-AUC (weighted) | 0.072 | 0.036 to 0.109 |
| Primary − ADA-style, PR-AUC (weighted) | 0.034 | 0.019 to 0.061 |
| Women − men, ROC-AUC | 0.089 | 0.022 to 0.154 |
| Women − men, PR-AUC | 0.077 | 0.025 to 0.150 |
| Women − men, calibration slope | 0.34 | 0.06 to 0.63 |

- **Calibration by sex:** O/E was 1.14 for women (0.90–1.37) and 0.83 for men (0.64–1.02).
- **In men:** predicted risks in the top fifth averaged 11.7 %, against 8.2 % observed, so risks were too high at the top. In the second fifth they were 1.3 % predicted against 2.8 % observed, so too low at the low end.
- **In women:** the top fifth was slightly underestimated (10.9 % predicted vs 13.5 % observed).
- **Caution:** this rests on 67 male and 90 female positives, and it was looked for only after the results were seen.

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
- **2017-2018 was not untouched.** Before the analysis plan, the v1 pipeline scored XGBoost on
  2017-2018 once (commit `df1b305`: PR-AUC 0.101, ROC-AUC 0.776, prevalence 3.8 %, v1 rules). No
  selection rule used it, but the cycle's prevalence and one model's performance were known.
- **Test item mapping.** The 2017 alcohol questionnaire was redesigned, so the three alcohol
  items are missing on the test and were imputed (this affects EBM / B only). Other 2017-2018
  items keep their 2011-2016 names, and their codes were assumed unchanged without re-checking
  every codebook.
- **One test cycle, small subgroups.** 157 test positives; only one race/ethnicity group
  reaches 30, and the men-vs-women difference rests on 67 male positives.
- **Label based on HbA1c only.** The main label is HbA1c ≥ 6.5 % from a single measurement, with
  no confirmatory test. HbA1c misses some cases that fasting glucose or an OGTT would find, and
  differs by race/ethnicity and with anaemia or haemoglobin variants. HbA1c-or-FPG on the
  fasting subsample is a sensitivity analysis only.
