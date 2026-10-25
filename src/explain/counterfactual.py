"""DiCE counterfactuals with immutable features. Outputs are model explanations, not causal advice."""
from __future__ import annotations

import dice_ml
import pandas as pd


def make_dice(pipe, train: pd.DataFrame, target: str, continuous: list[str]):
    """`pipe` is the full fitted pipeline (takes raw features); `train` includes the target column."""
    data = dice_ml.Data(dataframe=train, continuous_features=continuous, outcome_name=target)
    model = dice_ml.Model(model=pipe, backend="sklearn")
    return dice_ml.Dice(data, model, method="random")


def counterfactuals(dice, query: pd.DataFrame, immutable: list[str], total_cfs: int = 3,
                    permitted_range: dict | None = None):
    varying = [c for c in query.columns if c not in immutable]
    return dice.generate_counterfactuals(
        query, total_CFs=total_cfs, desired_class="opposite",
        features_to_vary=varying, permitted_range=permitted_range or {})
