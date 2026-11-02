# Model card: GlucoSense primary model

L2 logistic regression on set A (10 routine inputs, one-hot categoricals), refit on NHANES
2011-2016, frozen at the commit tagged `pre-test` (`6795066`) and tested once on 2017-2018. Not
recalibrated. Portable copy for the demo: `models/primary_logreg_A.json`.

## Intended use

- **Purpose:** research and teaching on screening for undiagnosed diabetes in US adults, from
  information available without a blood test.
- **Population:** non-pregnant adults aged 20 or over, without diagnosed diabetes, not on
  insulin or diabetes pills, as in NHANES.
- **Output:** a probability of HbA1c ≥ 6.5 % (undiagnosed diabetes). The decision curve was
  studied at thresholds of 1-5 %.
- **Not intended for:** diagnosis, treatment decisions, people with known diabetes, pregnancy,
  children, or populations outside the US without local validation. It is **not a medical
  device** and has no regulatory clearance.

## Inputs

age, sex, BMI, waist circumference, height, mean systolic and diastolic blood pressure, family
history of diabetes, ever smoked 100 cigarettes, and race/ethnicity (NHANES RIDRETH3). Four
interaction terms are derived from these inputs (age × BMI, BMI/age, waist/height, pulse
pressure). The portable copy needs every numeric input. Missing categorical answers take the
training mode.

## Data

| | Development | Temporal test |
|---|---|---|
| NHANES cycles | 2011-2012, 2013-2014, 2015-2016 | 2017-2018 |
| n (positives) | 13,189 (410) | 4,095 (157) |
| Prevalence, unweighted / weighted | 3.1 % / 2.1 % | 3.8 % / 2.5 % |

Rules for the sample, the labels and the features are in `docs/analysis_plan.md`, sections 1-3.
**The 2017-2018 cycle was not untouched:** an older v1 model scored it once before the plan
existed. No v2 decision used that result.

## Performance

Bootstrap 95 % intervals (1000 resamples, seed 42). Development values come from pooled nested-CV
out-of-fold predictions; the mean across the 15 folds is ROC-AUC 0.805 and PR-AUC 0.104.

| Metric | Development (out-of-fold) | **Test 2017-2018** |
|---|---|---|
| ROC-AUC | 0.805 (0.789-0.821) | **0.780 (0.749-0.811)** |
| PR-AUC | 0.098 (0.085-0.117) | **0.107 (0.085-0.139)** |
| ROC-AUC, survey-weighted | 0.844 | **0.799 (0.751-0.846)** |
| PR-AUC, survey-weighted | 0.088 | **0.082 (0.063-0.114)** |
| Calibration slope | 1.00 (0.93-1.09) | **0.89 (0.76-1.02)** |
| Calibration intercept | 0.01 (fold mean) | **−0.33 (−0.74 to 0.05)** |
| O/E | 1.00 (0.91-1.10) | **0.98 (0.84-1.13)** |
| Brier | 0.029 (fold mean) | **0.036 (0.031-0.041)** |

- **ADA-style comparator on the test:** ROC-AUC 0.717 and PR-AUC 0.072. The primary model is
  ahead by ROC-AUC +0.063 (0.040-0.087) and PR-AUC +0.035 (0.023-0.057). This paired difference
  is post-test and exploratory.
- **Decision curve on the test:** net benefit is above treat-all at every threshold from 1 % to
  5 %. At 3 % the difference is +0.0107 (0.0080-0.0134).

**Subgroups.** The sex rows are exploratory. The test rows were looked at only after scoring.

| Group | Dev ROC-AUC | Test ROC-AUC | Test calibration slope | Test O/E |
|---|---|---|---|---|
| Women | 0.803 (0.774-0.828) | 0.819 (0.783-0.856) | 1.04 | 1.14 (0.90-1.37) |
| Men | 0.805 (0.783-0.828) | 0.730 (0.674-0.782) | 0.70 (0.52-0.92) | 0.83 (0.64-1.02) |
| Women − men | −0.002 (−0.036 to 0.034) | **+0.089 (0.022 to 0.154)** | | |
| Non-Hispanic Black | | 0.743 (0.678-0.806) | 0.73 | 1.07 (0.81-1.33) |

The other race/ethnicity groups had fewer than 30 positives in the test, so they are flagged
unreliable. In development, all five reliable groups had an O/E interval that included 1.

## Limitations

- **One country, one survey design:** the model is validated on one later NHANES cycle only, and
  there is no external validation.
- **The label is a single HbA1c measurement**, with no repeat test and no FPG or OGTT. HbA1c is
  biased in some groups, for example with anaemia or haemoglobin variants.
- **Discrimination fell over time** (0.805 → 0.780). The likely reasons are case-mix drift and
  the 2015 change of glucose analyser. The 2015-2016 LOCO showed the same fall.
- **Sex gap on the test only:** the model ranks men worse and is too extreme for them (slope
  0.70). This rests on 67 male positives and was not seen in development.
- **Small subgroups:** too few test positives to check calibration for most race/ethnicity groups.
- **Pre-plan look at 2017-2018** by the v1 model, disclosed above.
- **The bootstrap ignores the survey design:** rows are resampled independently, ignoring strata
  and PSUs, so the weighted intervals are approximate.

## Ethical considerations

- **Race/ethnicity is an input.** It was pre-specified in set A, as in common risk scores. Race here is a social category, a proxy for unmeasured exposures and access, not
  biology. Anyone deploying the model should weigh the risk of reinforcing inequities and check
  calibration per group.
- **False reassurance is the main harm:** a low score does not rule out diabetes. The model is
  meant to add to blood testing, never to replace it, and testing guidelines (for example,
  USPSTF and ADA) still apply.
- **Uneven performance by sex on the test.** A use that relies on the score alone could miss more
  cases in men.
- **Privacy:** the data are public-use and de-identified. No NHANES records are redistributed, and
  the portable model holds no individual data.

## Provenance

Every hash is in `results/test_2017_2018/run_log.json`, and the reporting checklist is in
`docs/tripod_ai_checklist.md`. The license is MIT, for the code only (see `LICENSE`).
