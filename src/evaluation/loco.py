"""Leave-one-cycle-out (LOCO) on the development cycles: tune on the other cycles with inner CV,
score once on the held-out cycle. A sensitivity check for cycle effects (the 2015-2016 glucose
instrument and pill-question changes). Unlike the temporal test, training may include later
cycles, so it is not a forward-in-time validation."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.evaluation.metrics import calibration_stats, summarize
from src.models.models import RAW_INPUT_MODELS, get_model
from src.preprocessing.pipeline import build_model_pipeline


def loco(df: pd.DataFrame, feats: list[str], categorical: list[str], target: str,
         model_name: str, strategy: str, cfg: dict, n_iter: int) -> pd.DataFrame:
    seed, cv = cfg["seed"], cfg["cv"]
    rows = []
    for held_out in sorted(df.cycle.unique()):
        train, test = df[df.cycle != held_out], df[df.cycle == held_out]
        model, space = get_model(model_name, seed)
        t0 = time.perf_counter()
        if model_name in RAW_INPUT_MODELS:
            search, best = model.fit(train[feats], train[target]), {}
        else:
            pipe = build_model_pipeline(train[feats], model, categorical, strategy,
                                        cfg["preprocessing"]["knn_neighbors"],
                                        cfg["features"]["interactions"], seed)
            search = RandomizedSearchCV(
                pipe, space, n_iter=n_iter, scoring=cv["scoring"], random_state=seed, n_jobs=1,
                cv=StratifiedKFold(cv["inner_folds"], shuffle=True, random_state=seed), refit=True)
            search.fit(train[feats], train[target])
            best = search.best_params_
        y, p = test[target].to_numpy(), search.predict_proba(test[feats])[:, 1]
        cal = calibration_stats(y, p)
        rows.append({"model": model_name, "strategy": strategy, "held_out_cycle": held_out,
                     "n_iter": n_iter, "best_params": str(best), **summarize(y, p),
                     "calibration_slope": cal["slope"], "calibration_intercept": cal["intercept"],
                     "citl_difference": cal["citl_difference"],
                     "citl_ratio_observed_expected": cal["citl_ratio_observed_expected"],
                     "n_pos": int(y.sum()), "n": len(y),
                     "train_prevalence": float(np.mean(train[target])),
                     "fit_seconds": time.perf_counter() - t0})
    return pd.DataFrame(rows)
