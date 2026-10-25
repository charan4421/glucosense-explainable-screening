# GlucoSense-AI

Calibrated, explainable screening for **undiagnosed diabetes** from non-laboratory features.

- **Main data:** NHANES 2011-2018 (4 cycles). Lab values and survey weights are used only to build labels and prevalence estimates, never as features.
  - **Analysis sample:** adults >= 20 y with HbA1c measured and no self-reported diagnosis (DIQ010 not 1). Excluded: refused/don't know on DIQ010 (7, 9) and anyone taking insulin (DIQ050) or diabetes pills (DIQ070) without a diagnosis. Borderline (DIQ010 = 3) is kept.
  - **Main label** `undiagnosed_diabetes`: HbA1c >= 6.5 %.
  - **Sensitivity label** `undiagnosed_diabetes_a1c_or_fpg` (sensitivity analysis only): HbA1c >= 6.5 % or fasting plasma glucose >= 126 mg/dL, defined only for the fasting subsample (missing otherwise).
  - **Prevalence** is reported unweighted and survey-weighted, per cycle and pooled: WTMEC2YR for the main label, WTSAF2YR for the sensitivity label, each divided by the number of cycles when pooling. Current build: main label 3.3 % unweighted / 2.1 % weighted (pooled, n = 17,493).
- **External validation:** BRFSS (overlapping features only).
- **Validation design:** nested 5-fold CV, plus temporal split (train on earlier NHANES cycles, test on the latest).
- **Models:** XGBoost, PyTorch MLP, TabNet.
- **Imbalance strategies compared:** none, class weighting, SMOTE-NC (always applied inside CV folds).
- **Metrics:** PR-AUC (primary), ROC-AUC, Brier score, calibration curve, Decision Curve Analysis.
- **Explainability:** SHAP (global + local), LIME consistency check, DiCE counterfactuals with immutable features.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.data.nhanes --download                 # downloads XPT files from CDC
python scripts/run_experiment.py --config configs/config.yaml
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
