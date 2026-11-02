# TRIPOD+AI checklist

This checklist is for a model that is both developed and evaluated (Collins GS et al., *BMJ*
2024;385:e078378). The item wording below is shortened, so check it against the official
checklist before submitting.

- **Where:** the place in this repository where the item is reported. "Manuscript" means the
  paper has to report it and the repository does not cover it yet.
- **Plan:** `docs/analysis_plan.md`.
- **Card:** `docs/model_card.md`.

| # | Item (shortened) | Status | Where |
|---|---|---|---|
| 1 | Title: development and evaluation, population, outcome | Manuscript | README title is a draft |
| 2 | Abstract (TRIPOD+AI for Abstracts) | Manuscript | |
| 3a | Background: healthcare context, rationale | Partly | README introduction |
| 3b | Target population and intended purpose in the care pathway | Done | Card, "Intended use" |
| 3c | Known health inequalities between groups | Partly | Card, "Ethical considerations" |
| 4 | Objectives (development and evaluation) | Done | Plan, sections 1 and 9 |
| 5a | Data sources, and why they are representative | Done | Plan, section 1; NHANES survey design |
| 5b | Dates of data collection | Done | NHANES cycles 2011-2018 |
| 6a | Setting (community, survey) | Done | Card, "Data" |
| 6b | Eligibility criteria | Done | Plan, section 2 |
| 6c | Treatments received, if relevant | Done | Insulin and pill users excluded (plan, section 2) |
| 7 | Data preparation, quality checks | Done | Plan, sections 1-3 (XPT zero fix, skip patterns, caps); `src/data/derive.py` |
| 8a | Outcome definition and timing | Done | HbA1c ≥ 6.5 % at the exam (plan, section 2) |
| 8b | Qualifications and demographics of outcome assessors | N/A | Laboratory measurement (NHANES lab protocol) |
| 8c | Blinding of outcome assessment | Done | Lab assay, independent of the predictors |
| 9a | Choice of predictors | Done | Plan, section 3 (sets A-D, exclusions) |
| 9b | Predictor definitions and measurement | Done | Plan, section 3; `docs/feature_audit.md` |
| 9c | Predictor assessors | N/A | NHANES exam and interview protocols |
| 10 | Sample size and its justification | **Missing** | Only the fixed n and events are reported. A formal sample-size justification (for example Riley et al.) is not done |
| 11 | Missing data and how it was handled | Done | Plan, sections 3-4 (skip-pattern zeros, KNN and mode imputation inside CV) |
| 12a | Data used for development and evaluation | Done | Plan, sections 4 and 9 |
| 12b | Handling of predictors (transformations, scaling) | Done | Plan, section 4; `src/preprocessing/pipeline.py` |
| 12c | Model type, building, hyperparameter tuning | Done | Plan, section 4 (nested CV 5×3, inner 3-fold, n_iter 5) |
| 12d | Heterogeneity across clusters (cycles) | Done | Plan, section 7 (LOCO) |
| 12e | Performance measures, with CIs | Done | Plan, sections 5 and 9 |
| 12f | Model updating or recalibration | Done | Section 9 intercept recalibration (secondary) |
| 12g | Methods for comparing with an existing model | Done | ADA-style comparator (2026-11-02 deviation) |
| 13 | Class imbalance methods | Done | Plan, section 4 (rebalancing variants on set A; primary has none) |
| 14 | Fairness approaches | Done | Subgroups by sex and race/ethnicity, calibration by group (plan, sections 5 and 9; Deviations) |
| 15 | Model output and any thresholds | Done | Probability; decision curve 1-5 % (card) |
| 16 | Differences between development and evaluation data | Done | Plan, section 9 (item mapping, FPG instrument, alcohol items) |
| 17 | Ethical approval | Partly | NHANES protocols are approved by the NCHS Ethics Review Board; secondary use of public data. State this in the manuscript |
| 18a | Funding | Manuscript | |
| 18b | Conflicts of interest | Manuscript | |
| 18c | Protocol | Done | Plan, committed before the runs, with dated deviations |
| 18d | Registration | **Missing** | Not registered (the commit history is the only timestamp) |
| 18e | Data availability | Done | NHANES public files; `LICENSE` notice; `src/data/nhanes.py` downloads them |
| 18f | Code availability | Done | This repository (MIT) |
| 19 | Patient and public involvement | **Missing** | None, which must be stated |
| 20a | Participant flow | Partly | Exclusion counts in plan section 2; a flow diagram is still needed |
| 20b | Participant characteristics, by data set | **Missing** | A "Table 1" (dev vs test) has not been produced yet |
| 20c | Comparison of predictor distributions, dev vs test | **Missing** | Same as 20b |
| 21 | Participants and outcome events in each analysis | Done | README results; card |
| 22 | Full model (coefficients) for reuse | Done | `models/primary_logreg_A.json` (coefficients, scaling, levels) |
| 23a | Performance with CIs | Done | README results; `results/test_2017_2018/` |
| 23b | Performance across subgroups | Done | README; `test_subgroups_*.csv`; post-test exploratory |
| 24 | Model updating results | Done | Intercept-recalibrated row (README) |
| 25 | Interpretation, overall | Partly | README post-test findings |
| 26 | Limitations, including bias and generalisability | Done | README "Limitations"; card |
| 27a | Handling of poor-quality or missing inputs at use | Done | Card, "Inputs" (portable model needs complete numerics) |
| 27b | User interaction and expertise needed | Partly | Card, "Intended use"; demo only |
| 27c | Future research, applicability | Partly | Next validation on NHANES 2021-2023, to be pre-registered |

**Still to do before submission:** item 10 (sample size), 18d (registration), 19 (patient and
public involvement statement), 20a-c (flow diagram and Table 1), and the manuscript-only items.
