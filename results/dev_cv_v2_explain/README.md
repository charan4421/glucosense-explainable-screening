Explanation analysis, development data 2011-2016 only (docs/analysis_plan.md, deviation 2026-11-02 B). Exploratory.

Counterfactuals (dice_*.csv) explain the model: they show which input changes would move the model's predicted risk below 3 %. They are not clinical advice, not causal claims and not recommendations for any person.

DiCE implementation detail: numeric inputs whose observed training values are all whole numbers (flags, counts) are given 0 decimal places, so counterfactual values stay whole.
{
  "patients_per_cell": 17,
  "background": 100,
  "dice_patients": 20,
  "cfs_per_patient": 3,
  "threshold": 0.03,
  "immutable": [
    "age",
    "sex",
    "height_cm",
    "race_eth",
    "family_history_diabetes",
    "smoker"
  ]
}
