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
  same tuning procedure, and scored **once** on 2017-2018. There is no retuning, recalibration or
  feature change after seeing that cycle.
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
