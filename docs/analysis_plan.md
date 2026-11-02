# Analysis plan: development phase (NHANES 2011-2016)

Committed **before any model in this plan was run**. Any change after this commit is recorded
under "Deviations" at the end, with its date and reason.

The 2017-2018 cycle is not read in this phase. It is reserved for a single temporal test of the
model chosen here (section 9).

## 1. Data

- **Cycles:** 2011-2012, 2013-2014, 2015-2016.
- **Build:** `python -m src.data.nhanes` writes `data/processed/nhanes_dev.parquet`. The build
  refuses the test cycle.
- **Analysis sample:** n = **13,189**, with **410** main-label positives (3.1% unweighted, 2.1%
  weighted with WTMEC2YR/3).
- **XPT zero bug fixed:** pandas reads SAS XPORT zeros as 5.4e-79, so they are restored to 0 at
  read time. This changed the fasting subsample (zero fasting weights had been counted as
  fasting) and every `== 0` skip gate.

## 2. Sample and labels (applied by rule)

| Rule | 2011-12 | 2013-14 | 2015-16 |
|---|---|---|---|
| Adults ≥ 20, HbA1c measured, DIQ010 not 1/7/9 (borderline kept), no insulin (DIQ050) or diabetes pills (DIQ070) | | | |
| **Exclude pregnant at exam (RIDEXPRG = 1)** | −50 | −59 | −57 |
| n analysed | 4,288 | 4,607 | 4,294 |
| **Main label:** HbA1c ≥ 6.5 %, unchanged (positives) | 152 | 131 | 127 |
| **Sensitivity label:** HbA1c ≥ 6.5 % or FPG ≥ 126, fasting subsample (WTSAF2YR > 0); positives before → after FPG back-adjustment | 122 → 122 | 112 → 112 | **125 → 99** |

- **FPG back-adjustment (sensitivity label only):** 2015-2016 FPG was measured on a Roche Cobas
  C311, which reads about 2% higher. It is mapped back to the earlier C501 scale with CDC's
  backward equation (GLU_I analytic note): **C501 = 0.9776 × C311 + 0.4994**.
  - **26 positives change**, all in 2015-2016.
  - The weighted sensitivity prevalence becomes 4.1% / 3.9% / 4.0% across the three cycles.
- **The main label is the primary outcome.** The sensitivity label is used only for a secondary
  analysis.

## 3. Features

Sets are nested (A ⊂ B ⊂ C) and defined in `configs/config.yaml`. They are derived in
`src/data/derive.py` from the variables in `docs/feature_audit.md`, using the CDC codebook codes.

- **A (10):**
  - age, sex, BMI, waist, height
  - mean systolic and diastolic BP
  - family history of diabetes (MCQ300C)
  - ever smoked 100 cigarettes (SMQ020)
  - race/ethnicity (RIDRETH3)
- **B = A + 30:**
  - **Physical activity:** MET-minutes per week (GPAQ weights: vigorous 8, moderate and
    transport 4; PAQ605-PAD675) and sedentary minutes per day (PAD680).
  - **Sleep:** sleep hours (SLD010H in 2011-2014, SLD012 in 2015-2016, capped at 12) and told a
    doctor about trouble sleeping (SLQ050).
  - **Smoking:** status (never/former/current) and cigarettes per day.
  - **Alcohol:** status (lifetime abstainer/former/current), drinks per week (ALQ120Q/U ×
    ALQ130, capped at 15 per day) and binge days per year (ALQ141Q/U).
  - **Diet quality:**
    - self-rated diet (DBQ700)
    - meals out and fast-food meals per week
    - ready-to-eat and frozen meals per 30 days (capped at 90)
    - from reliable day-1 recalls only (DR1DRSTZ = 1): energy, fibre per 1,000 kcal, and sugar,
      saturated fat and carbohydrate as % of energy
  - **Hypertension:** told high BP (BPQ020) and now taking BP medication (BPQ050A).
  - **Gestational diabetes history:**
    - RHQ162; "borderline" is coded 0
    - baby ≥ 9 lb (RHQ172)
    - 0 for men and never-pregnant women
  - **Weight history**, from self-report, in kg:
    - change over 1 year
    - change over 10 years (+ not-applicable flag under 36)
    - gain since age 25 (+ not-applicable flag under 27)
    - below-maximum weight
    - tried to lose weight
- **C = B + 3:** education (DMDEDUC2), income-to-poverty ratio (INDFMPIR) and adult food
  security (FSDAD).
- **D = best of B/C + 4 (sensitivity only, never primary):**
  - health insurance (HIQ011)
  - routine place of care (HUQ030)
  - healthcare visits (HUQ050/HUQ051 mapped to common bins 0, 1, 2-3, 4-9, 10-12, 13+)
  - overnight hospital stay (HUQ071)

**Excluded by decision:**
- prediabetes and at-risk items (DIQ160/170/172/180)
- doctor's advice and behaviour change (MCQ365/370)
- complications (MCQ160B-F, KIQ022) and nocturia (KIQ480)
- pregnancy as a feature (RIDEXPRG is used only for the exclusion)
- cholesterol diagnosis and medication (BPQ080/090D)
- WHQ060 (intentional weight change)
- all lab values and labels (enforced by `feature_set`; raw `gate_*` columns are refused too)

**Could not be mapped for all three cycles, so not used:**
- SLQ030, SLQ040 and SLQ120 (2015-2016 only) and SLQ060 (2011-2014 only)
- grip strength (MGX, 2011-2014 only)

**In the audit but left out (redundant or outside the requested groups):**
- TV/computer hours (PAQ710/715; overlap with PAD680)
- age started smoking and time since quitting
- ALQ151
- weight perception and age at heaviest weight
- milk, sodium and total fat
- BPQ030 and BPD035
- household food security and SNAP
- marital status, birthplace, language, and other reproductive items

**Skip patterns:**
- A "no" to a gate question gives the follow-up a real zero **before imputation**: 0 MET
  minutes, 0 cigarettes, 0 drinks, 0 fast-food meals, no BP medication, no GDM.
- Where zero means "not asked" rather than "none", a `*_na` indicator is kept (weight 10 years
  ago and at age 25).
- `assert_no_structural_missing` runs on every build. A test fails if a skip-pattern feature
  reaches the imputer with structural missingness, checked both on synthetic rows and on the
  built data.
- Refused and don't-know answers are left missing and imputed.

## 4. Models and validation design

- **Main models:**
  - L2 logistic regression (C tuned)
  - XGBoost with no rebalancing
  - Explainable Boosting Machine (interpret-core; 8 outer bags, 2 jobs (see Deviations); interactions,
    learning rate, leaves and minimum leaf size tuned)
  - MLP

  All four are run on **sets A, B and C**.
- **Set A only:**
  - XGBoost and MLP with class weighting and with SMOTE-NC
  - TabNet with no rebalancing and with SMOTE-NC

  They were dominated and slow in the earlier run, and they are run last.
- **Set D:** the four main models. It is reported but never primary.
- **Validation:** nested cross-validation.
  - **Outer:** 5-fold stratified, **repeated 3 times** (15 outer folds, seed 42).
  - **Inner:** 3-fold stratified, tuning by randomised search with **n_iter = 5 for every
    model** (the same budget) and PR-AUC scoring.
  - Folds depend only on the outcome and the seed, and every set uses the same rows, so **all
    models and sets are scored on identical folds**.
- **Preprocessing:** fitted inside each training fold only.
  - interaction terms
  - KNN imputation (k = 5)
  - robust scaling for numeric features
  - most-frequent imputation and ordinal encoding for categorical features
  - SMOTE-NC inside the fold, where it is used
  - KNN distance chunks are capped at 128 MB (`sklearn.set_config(working_memory=128)`). This
    changes memory use only, not results.
- **Out-of-fold predictions:** averaged over the 3 repeats. They are used for calibration plots,
  decision curves and subgroups.

## 5. Metrics and reporting

- **Per outer fold:**
  - PR-AUC (primary)
  - ROC-AUC
  - Brier score
  - calibration slope and intercept (logistic regression on logit p)
  - calibration-in-the-large (observed − predicted, and O/E)
  - n and n positive
- **Summary:** mean ± SD over the 15 outer folds, in `results/dev_cv_v2_summary.csv`. **SDs
  across repeated folds are not independent** (training sets overlap), so they understate the
  true uncertainty. They describe spread; they are not standard errors.
- **Paired differences:**
  - Model pairs within each set, and set pairs within each model.
  - Metrics: PR-AUC, ROC-AUC and Brier.
  - Computed on identical folds with the **Nadeau-Bengio corrected 95% interval**:
    variance × (1/15 + 1/4), t with 14 df.
  - Output: `results/dev_cv_v2_paired.csv`.
- **Read PR-AUC against prevalence** (about 3.1% in every fold).

## 6. Selection rules (pre-specified)

The rules are implemented in `src/evaluation/selection.py` and tested on a toy summary.

1. **Primary set:**
   - For each of A, B and C, take its best main model by mean PR-AUC.
   - Let *best* be the highest of these, and *SD* its SD across the 15 outer folds.
   - The primary set is the **smallest set** (A < B < C) whose best mean PR-AUC ≥ best − SD.
2. **Primary model:**
   - On the primary set, take the best main model's mean − SD.
   - The primary model is the **simplest** main model whose mean PR-AUC reaches that threshold.
   - Simplicity order: logistic regression < EBM < XGBoost < MLP.
3. **Base of set D:** whichever of B and C has the higher mean PR-AUC averaged over the four main
   models.
4. **Rebalancing, TabNet and set D are never selected.** They are reported as sensitivity
   analyses.
5. **The Nadeau-Bengio intervals are reported alongside the rules but do not change them.**

## 7. Leave-one-cycle-out sensitivity

- **Why:** the 2015-2016 glucose instrument and the pill-question universe changed.
- **What:**
  - The four main models on the primary set.
  - Each development cycle is held out in turn.
  - Tuning is done on the other two cycles (inner 3-fold, n_iter = 5), and the model is scored
    once on the held-out cycle.
- **Reported per held-out cycle:** PR-AUC (with that cycle's prevalence), ROC-AUC, Brier,
  calibration slope and intercept, and O/E.
- **Output:** `results/dev_cv_v2_loco.csv`.
- **Caveat:** training can include later cycles, so this is not forward-in-time validation.

## 8. Run order and checkpoints

`python scripts/run_dev.py` runs the stages in this order:
1. Main grid (logistic regression, XGBoost, MLP, then EBM, × A, B, C)
2. Set D
3. Leave-one-cycle-out
4. Set-A extras, with TabNet last

Each combination is saved to `results/dev_cv_v2/` as soon as it finishes. The file carries a hash
of the data file, the CV and preprocessing settings, the resolved feature list, the seed and
n_iter. A restart skips a combination only if the hash matches.

## 9. Fixed now for the 2017-2018 temporal test

- **Sample and labels:** the same exclusions (including pregnancy at the exam), the same main
  label (HbA1c ≥ 6.5 %) and the same derived-variable definitions and caps.
- **FPG rule:** FPG measured on the Cobas C311 is back-adjusted with the same equation, for the
  sensitivity label only. Which instrument 2017-2018 used will be read from its documentation at
  test time, and the rule applied mechanically.
- **The model:** the primary model on the primary set is refit on all 2011-2016 data with the
  same tuning procedure, and scored **once** on 2017-2018. After the encoding fix (Deviations,
  2026-11-02) the selection rules, applied again, still pick logistic regression on set A; the
  test model is that model **with one-hot encoded categorical inputs**, not the ordinal one. There is no retuning or feature change
  after seeing that cycle, and the **primary test result is uncalibrated**: the 2011-2016 model's
  predictions exactly as they come out.
- **What the test reports** (added 2026-11-02, see Deviations):
  - **Unweighted (primary):** PR-AUC (with the 2017-2018 prevalence), ROC-AUC, Brier,
    calibration slope and intercept, O/E, and the decision curve over the 1-5 % window, with
    bootstrap 95 % intervals (1000 resamples, seed 42).
  - **Survey-weighted (secondary):** ROC-AUC and PR-AUC weighted with the 2017-2018 2-year MEC
    weight (WTMEC2YR, one cycle, so not divided), with the weighted prevalence and the same
    bootstrap. Like the development analysis, the intervals ignore strata and PSUs.
  - **Intercept-recalibrated (secondary):** the intercept is re-estimated on 2017-2018 with the
    slope fixed at 1 (logit p as offset). This uses the test outcomes, so it shows what a local
    intercept update would give, not an independent estimate. It changes calibration-in-the-large,
    O/E, Brier and net benefit; ROC-AUC and PR-AUC do not change, because the ranking does not.
    Both versions are reported side by side, with the uncalibrated one first.
- **Expected performance:** test performance may well be lower than the development nested-CV
  estimate. The closest development analogue is leave-one-cycle-out with 2015-2016 held out,
  the latest and weakest cycle for every model. For the one-hot logistic regression on set A
  that gave PR-AUC 0.079 (prevalence 3.0 %), ROC-AUC 0.773, O/E 0.79 and calibration intercept
  −0.73 (risk overpredicted), against nested-CV PR-AUC 0.104 and ROC-AUC 0.805. (The ordinal
  model gave 0.073, 0.760, 0.84 and −0.61, against 0.094 and 0.783.) A drop of that size is
  read as expected temporal drift, not as a failure of the analysis.
- **If an item can't be reproduced in 2017-2018 with the same definition:**
  - This is expected for alcohol, whose questionnaire was redesigned in 2017. That is general
    knowledge, not yet verified, because the cycle was not opened.
  - The mapping will be documented before scoring.
  - An item that cannot be mapped is left missing and imputed by the training-fitted imputer,
    and reported as a limitation.

## 10. Known limitations

- The tuning budget is small (n_iter = 5) and the same for every model. The EBM uses 8 rather
  than 14 outer bags for compute reasons.
- Fold SDs are not independent (section 5).
- Self-reported behaviour has recall bias, and a single 24-h dietary recall is noisy for one
  person.
- Physical activity, diet and weight history are partly self-reported current behaviour, which
  people with undiagnosed diabetes may report differently.

## Deviations

- **2026-11-01: EBM parallel jobs reduced from 4 to 2 (`n_jobs`), because of memory.**
  - **Why:** the first run (started 22:24) was stopped by the system at 23:05 for low memory,
    during XGBoost on set B.
  - **Effect on results:** none expected. `n_jobs` only sets how many outer bags are fitted in
    parallel, each bag is seeded from `random_state`, and nothing else changes. EBM had not
    started when the change was made.
  - **Kept:** the four combinations finished before the stop, saved with their hashes. The
    hashes cover data, CV, preprocessing and features, not EBM's `n_jobs`, so these
    combinations remain valid:
    - logistic regression / A, B, C
    - XGBoost / A
  - **Restart:** after a reboot, and only once at least 3 GB of memory is free.

- **2026-11-02: survey-weighted discrimination (sensitivity), and bootstrap bands for the
  decision curve and subgroup tables.** Added after the development run had finished, written
  before any of these analyses was run. Development data (2011-2016) only; 2017-2018 is not read.
  - **Why:** until now the survey weights were used only for prevalence (`prevalence_report`,
    WTMEC2YR for the main label, divided by the number of pooled cycles). Every model metric was
    unweighted, so none of them describes the US adult population. The decision curve and the
    subgroup tables had point estimates only.
  - **Weighted sensitivity analysis:**
    - **Model:** the primary model on the primary set, from the selection rules
      (`results/dev_cv_v2_selection.json`). No refit and no retuning: the existing out-of-fold
      predictions, averaged over the 3 repeats, are scored.
    - **Weights:** WTMEC2YR / 3 pooled, and WTMEC2YR within each cycle. The weights enter only the
      metric (`sample_weight`), never training.
    - **Metrics:** weighted ROC-AUC and weighted PR-AUC (average precision), each next to the
      unweighted value on the same pooled out-of-fold predictions. Weighted PR-AUC is read
      against the weighted prevalence (about 2.1 %), not the unweighted 3.1 %.
    - **Intervals:** percentile bootstrap, 1000 resamples of persons (seed 42), with each person
      keeping their weight. This **ignores the NHANES design** (strata SDMVSTRA and PSUs
      SDMVPSU are not in the development file, and adding them would change the data hash), so
      the intervals are too narrow and are only approximate.
    - **The unweighted results stay primary.** The weighted results change no selection rule.
    - **Output:** `results/dev_cv_v2_weighted.csv`.
  - **Bootstrap bands:** 1000 resamples of persons, seed 42, percentile 95 % intervals.
    - **Decision curve:** the pre-specified 1-5 % threshold window. Net benefit of each plotted
      model, of treat-all, and of each model minus treat-all, with every model scored on the
      same resamples. Output `results/dev_cv_v2_dca.csv` / `.png`.
    - **Subgroups (sex, race/ethnicity):** PR-AUC, ROC-AUC, calibration slope and O/E, resampled
      within each subgroup. Groups with fewer than 30 positives are still flagged unreliable.
      Output `results/dev_cv_v2_subgroups_<model>__<strategy>__<set>.csv`.
  - **Ported to v2:** `scripts/dev_dca.py` and `scripts/dev_subgroups.py` read the v1 build
    (`data/processed/nhanes.parquet`, which also holds 2017-2018) and the v1 out-of-fold files.
    They now read `data/processed/nhanes_dev.parquet` and `reports/dev_cv_v2_oof/` and check the
    combination hash. The threshold window, subgroup variables and the 30-positive flag are
    unchanged.

- **2026-11-02: selection record fixed (reporting only, no result changes).**
  - **What was wrong:** `scripts/run_dev.py` merged the outputs of the set rule and the model rule
    into one flat dict. Both rules return `best_model`, `best_pr_auc`, `best_pr_auc_sd` and
    `threshold`, so the model rule's values overwrote the set rule's. The JSON committed with the
    v2 results showed `best_set: C` next to set A's model-rule numbers (0.0964, SD 0.0116,
    threshold 0.0848), and the set rule's own numbers were missing.
  - **Fix:** `selection_record` (`src/evaluation/selection.py`, tested) writes each rule under
    its own key (`set_rule`, `model_rule`), with `primary_set`, `primary_model` and `d_base` at
    the top level.
  - **Effect:** none on any result. The rules were always applied to the right numbers; only the
    record was wrong. The JSON was regenerated by rerunning `scripts/run_dev.py`, which skipped
    every combination on its hash; summary, paired and LOCO files came out byte-identical.
    Set rule: best EBM / C 0.1128, SD 0.0182, threshold 0.0947 → primary set A (best 0.0964).
    Model rule on A: best EBM 0.0964, SD 0.0116, threshold 0.0848 → logistic regression (0.0938).

- **2026-11-02: section 9 extended before 2017-2018 was opened.**
  - **Added:** survey-weighted ROC-AUC and PR-AUC as a secondary test result (unweighted stays
    primary); an intercept-recalibrated result reported next to the uncalibrated primary one; and
    the stated expectation that test performance may be lower than development, anchored on the
    2015-2016 leave-one-cycle-out result.
  - **Changed wording:** "no recalibration" now means the primary result is uncalibrated. The
    recalibrated result is a labelled secondary analysis that uses the test outcomes.
  - **Why:** the development analysis now reports weighted discrimination (deviation above) and
    the 2015-2016 held-out cycle showed overprediction, so both are fixed for the test before it
    is run. 2017-2018 has not been read.

- **2026-11-02: A. calibration across race/ethnicity (exploratory extension).** Written before
  any of it was run. Development data (2011-2016) only; 2017-2018 is not read.
  - **Question:** does a simple fix reduce the race/ethnicity O/E gap of the primary model
    (logistic regression, set A, out-of-fold: non-Hispanic White 0.59, non-Hispanic Asian 1.66,
    Mexican American 1.37)?
  - **Observation behind it:** in the current pipeline `race_eth` reaches the logistic regression
    as one ordinal code (`OrdinalEncoder` maps NHANES codes 1, 2, 3, 4, 6, 7 to 0-5), so the model
    fits a single slope over arbitrary group codes and cannot give each group its own baseline.
    Tree models and EBM can split the code, so this concerns the logistic regression only.
  - **Candidates (fixed now), all logistic regression on set A:**
    1. **Interactions:** `race_eth` one-hot (reference: non-Hispanic White) as main effects, plus
       `race_eth × BMI` and `race_eth × waist`. The interactions need the one-hot coding, and
       their main effects come with them.
    2. **Group-wise intercept recalibration** of the primary model, fitted on training folds
       only. In each outer training fold the model is tuned as usual; inner cross-validated
       predictions on that training fold (same 3 inner folds, tuned hyperparameters) give one
       intercept offset per race/ethnicity group (logit p as offset, slope fixed at 1). Groups
       with fewer than 30 positives in the training fold keep offset 0. The offsets are applied
       to the outer test fold's predictions.
    - **Descriptive only, not a candidate:** one-hot `race_eth` main effects without
      interactions, to show how much of any change comes from the encoding alone.
  - **Evaluation:** the same nested CV as the main grid (5-fold × 3 repeats, seed 42, inner
    3-fold, n_iter = 5, identical folds). Per-fold PR-AUC, ROC-AUC, Brier and calibration; O/E by
    group from the averaged out-of-fold predictions, with the same within-group bootstrap as the
    subgroup tables (1000 resamples, seed 42).
  - **Decision rule (fixed now):** a candidate is adopted only if **both** hold:
    - its O/E 95 % CI includes 1 for more race/ethnicity groups than the primary model's. The
      count runs over the five groups with ≥ 30 positives (Mexican American, other Hispanic,
      non-Hispanic White, non-Hispanic Black, non-Hispanic Asian); other/multiracial (10
      positives) is reported but not counted. The primary model's count is 2 of 5 (Black,
      other Hispanic).
    - its mean PR-AUC over the 15 folds is ≥ the primary model's mean − one SD
      (0.0938 − 0.0153 = 0.0785).

    If both candidates qualify, the one covering more groups is adopted, then the higher PR-AUC.
    Otherwise the result is reported as a negative result.
  - **Either way:** the primary model stays logistic regression on set A as selected. An adopted
    candidate is reported as an exploratory extension and is **not** applied in the 2017-2018
    test (section 9) unless a further deviation, written before that cycle is opened, says so.
    Using race/ethnicity to adjust predicted risk is contested; that is discussed with the result.
  - **Output:** `results/dev_cv_v2_calib_ext/` (per-fold results and out-of-fold predictions per
    candidate), `results/dev_cv_v2_calib_ext_summary.csv`, `_subgroups.csv`, `_decision.json`.

- **2026-11-02: B. explanation analysis (exploratory).** Written before any of it was run.
  Development data only. Models: logistic regression / A (primary) and EBM / B (secondary).
  - **Fitted models:** the 15 outer-fold models and the 3 leave-one-cycle-out models are refit
    with exactly the nested-CV and LOCO procedure (same folds, seeds and tuning). Each refit is
    checked against the committed per-fold PR-AUC; any mismatch is reported.
  - **Feature space:** explanations are of the whole pipeline on the raw input features (10 for
    A, 40 for B), not the internal interaction terms. Missing inputs are filled with the fold's
    own fitted imputation, so every explained value is an imputed input value. Categorical
    inputs are integer-coded for the explainers and decoded before prediction.
  - **SHAP vs LIME agreement:**
    - **Patients:** 204 = 17 per race/ethnicity × sex cell (12 cells), sampled at random within
      each cell (seed 42). Each patient is explained by the repeat-0 outer-fold model for which
      they were in the test fold (held-out explanations).
    - **SHAP:** permutation explainer, independent masker with 100 background rows from that
      model's training fold (seed 42), default evaluation budget.
    - **LIME:** `LimeTabularExplainer` defaults (quartile discretisation, 5000 samples), the same
      training fold as reference data, all features, `random_state` 42.
    - **Agreement per patient:** top-5 overlap (shared features among the top 5 by |attribution|,
      divided by 5) and Spearman correlation of |SHAP| with |LIME weight| across all features.
    - **Summary:** mean with percentile bootstrap 95 % CI over patients (1000 resamples), overall,
      by race/ethnicity and by sex.
  - **Stability of global SHAP importance** (mean |SHAP| over the same 204 patients for every
    model, so differences come from the models; some patients are in a model's training data):
    - across the 15 outer-fold models: the 105 pairwise Spearman correlations (mean, min, max);
    - across the 3 LOCO models: the 3 pairwise correlations;
    - bootstrap 95 % CI for the mean pairwise correlation, resampling the 204 patients
      (1000 resamples), and each feature's rank range across models.
  - **DiCE counterfactuals:**
    - **Patients:** 20 sampled at random (seed 42) from patients predicted ≥ 3 % by both models'
      repeat-0 held-out predictions. 3 % is the middle of the pre-specified 1-5 % window and is
      close to the prevalence. Each is explained by the same held-out fold model as above.
    - **Target:** predicted risk < 3 % (DiCE's default 0.5 cut is meaningless at 3 % prevalence).
      DiCE `random` method, 3 counterfactuals per patient, seed 42.
    - **Immutable:** age, sex, height_cm, race_eth, family_history_diabetes, smoker. Everything
      else in the set may vary within the training range.
    - **Metrics:**
      - validity: share of counterfactuals rescored < 3 % by the model;
      - proximity: mean over changed numeric features of |change| / training MAD;
      - sparsity: number of features changed;
      - plausibility: share with every changed value inside the training 1st-99th percentile
        **and** no change to a history item that no action can change (gdm_history, baby_9lb,
        htn_told, sleep_trouble_told, weight_change_1y_kg, weight_change_10y_kg and its `_na`
        flag, weight_gain_since_25_kg and its `_na` flag, and smoking_status moving to "never").
    - **Framing:** the counterfactuals explain the model. They are not clinical advice or causal
      claims, and every output file says so.
  - **Output:** `results/dev_cv_v2_explain/`.

- **2026-11-02: specification error fixed: categorical inputs one-hot encoded in logistic
  regression.** Written before any rerun. Found after seeing development results and before the
  test; development data only; 2017-2018 has not been read.
  - **Error:** every categorical input went through `OrdinalEncoder`, so logistic regression
    treated a nominal variable as one numeric column with a single slope. For race/ethnicity the
    codes are NHANES 1, 2, 3, 4, 6, 7 mapped to 0-5. In sets B-D the same applies to
    `smoking_status` and `drinking_status`, whose codes follow alphabetical order
    (current < former < never; current < former < lifetime abstainer).
  - **How it was found:** the development subgroup analysis (O/E by race/ethnicity: non-Hispanic
    White 0.59, non-Hispanic Asian 1.66, Mexican American 1.37), and the exploratory extension A,
    where one-hot race main effects alone brought the O/E 95 % CI to include 1 for all five
    reliable groups (PR-AUC 0.1039 vs 0.0938).
  - **Fix:** for logistic regression only, categorical inputs are one-hot encoded
    (`OneHotEncoder`, binary items kept as one 0/1 column, unseen levels ignored) after the same
    most-frequent imputation. Binary items (sex, family history, ever smoked) are unchanged in
    effect. Everything else (folds, tuning, budget, features, preprocessing) is unchanged.
  - **Scope:** tree models and EBM split the code, so they are not refit. The MLP also receives
    ordinal codes; it is never the primary model and is not refit, and this is reported as a
    limitation. TabNet uses its own embedding path and is not affected in the same way.
  - **Rerun:** logistic regression on sets A-D and in LOCO, with the encoding recorded in the
    checkpoint hash (only for one-hot models, so other combinations stay valid). The ordinal
    results are kept in `results/archive_v2_ordinal_logreg/`. **The pre-specified selection
    rules (section 6) are applied again unchanged, and the outcome is reported whether or not
    the pick changes.** If the base of set D or the primary set changes, the stages that depend
    on them rerun by rule.
  - **Downstream:** the primary-model analyses (weighted metrics, decision curve, subgroups,
    explanation analysis B including DiCE) are rerun for the resulting primary model. In DiCE the
    race/ethnicity input stays immutable; as explanations are on raw inputs, that holds every
    one-hot race column fixed.
  - **Extension A:** candidate (ii), group-wise intercept recalibration, is dropped. With the
    encoding fixed, adjusting risk by race/ethnicity after the fact is not needed. Its committed
    results stay as a record.
- **2026-11-02: test protocol, written before 2017-2018 is scored.** Nothing from the v2
  build of 2017-2018 has been scored or looked at. Only its file and column names were read
  (to check item mappings) and its glucose documentation (section 9 FPG rule).
  - **Disclosure: earlier use of 2017-2018.** Before this plan existed, the v1 pipeline ran one
    temporal test (commit `df1b305`, `results/temporal_test.csv`). It used XGBoost (no
    rebalancing) on set A, with v1 rules (no pregnancy exclusion, XPT zero bug), and gave n
    4,138, prevalence 3.8 %, PR-AUC 0.101 and ROC-AUC 0.776. So the cycle's approximate
    prevalence and one model's performance on it were known before the plan. No selection rule
    used them. The test below is the first and only scoring of the pre-specified v2 model, but
    not the first look at the cycle, and this is reported as a limitation.
  - **Additions, all secondary and labelled as such (requested after section 9 was written,
    added before scoring):**
    - Subgroup tables (sex, race/ethnicity) for the primary model, as in development: within-group
      bootstrap (1000 resamples, seed 42), with groups under 30 positives flagged unreliable.
    - Bootstrap 95 % intervals also for calibration slope, intercept and O/E, on the same
      resamples as PR-AUC, ROC-AUC and Brier.
    - **EBM on set B** (the secondary model of explanation analysis B), refit on all of 2011-2016
      with the same procedure, and reported with the same unweighted metrics, weighted ROC/PR-AUC
      and decision curve. It never replaces the primary model.
    - **The ADA-style score** (`src/models/ada_style.py`): fixed points from set A inputs, with
      the points→risk mapping fitted on 2011-2016. It is a comparator. It was not run in the v2
      development analysis, so its development reference is a nested CV on 2011-2016 with the
      same folds, run in the fit stage before any test data is loaded.
    - The intercept-recalibrated version (section 9) is reported for the primary model only. In
      its bootstrap the intercept is re-estimated in every resample.
  - **2017-2018 build:** `python -m src.data.nhanes --test` writes
    `data/processed/nhanes_test.parquet` with the same exclusions, labels, derived variables
    and caps. The development build still refuses the test cycle.
    - **FPG rule, read from the GLU_J documentation:** "The Cobas C311 analyzer was used in
      2017-2018 and 2015-2016". So the same backward equation is applied mechanically, for the
      sensitivity label only. The test does not report the sensitivity label.
  - **Item mapping (section 9):**
    - **Alcohol:** the 2017 questionnaire replaced ALQ101/110/120/141 with
      ALQ111/121/142, which have different definitions: "ever had a drink" instead of 12 drinks
      in a lifetime or a year, a categorical frequency, and binge at 4+ drinks for women.
      `drinks_per_week`, `binge_days_per_year` and `drinking_status` cannot be reproduced with
      the same definition. They are left missing and imputed by the training-fitted imputers.
      This affects EBM / B only, not set A or the ADA-style score.
    - **Sleep hours and healthcare visits:** these use the existing harmonisation (SLD012,
      HUQ051).
    - **Other items** keep their 2011-2016 names. Their codes are assumed unchanged, and not
      every 2017-2018 codebook was re-checked.
  - **Procedure:**
    - The working tree is clean and the commit is tagged `pre-test`.
    - `scripts/run_test.py --stage fit` refits the models on 2011-2016 only (no test data
      loaded) and records the hashes in the run log: the commit, the code tree, the config, the
      development and test data, the 2017-2018 raw files, and the fitted models.
    - `--stage score` loads 2017-2018 and scores **once**. It refuses to run if the tree is
      dirty, if the tag does not match the commit, if the hashes differ from the run log, or if
      the output folder already exists.
    - Nothing is rerun or edited after the results are seen. Anything that looks wrong is
      reported as a post-test finding.
    - Output: `results/test_2017_2018/`.
