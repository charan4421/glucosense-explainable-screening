"""Explanation analysis, exploratory (docs/analysis_plan.md, deviation 2026-11-02 B), 2011-2016 only:
SHAP vs LIME agreement, stability of global SHAP importance, DiCE counterfactuals.
Models: logistic regression / A (primary) and EBM / B (secondary).

    python scripts/dev_explain.py --stage fit,shap     # per run: refit + SHAP/LIME + stability
    python scripts/dev_explain.py --stage dice         # needs both runs fitted

Counterfactuals explain the model; they are not clinical advice or causal claims.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so `src` imports work

import dice_ml  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from scripts.dev_subgroups import load_dev_oof  # noqa: E402
from scripts.run_dev import load_dev  # noqa: E402
from src.data.nhanes import MAIN_LABEL  # noqa: E402
from src.evaluation.loco import loco  # noqa: E402
from src.evaluation.nested_cv import nested_cv  # noqa: E402
from src.evaluation.subgroups import RACE_ETH  # noqa: E402
from src.evaluation.temporal import feature_set  # noqa: E402
from src.explain.analysis import (  # noqa: E402
    NOTE,
    Codec,
    agreement,
    bootstrap_mean,
    cf_metrics,
    imputed_raw,
    lime_weights,
    shap_values,
    stability,
)

OUT = Path("results/dev_cv_v2_explain")
CACHE = Path("reports/explain_cache")
RUNS = {"logreg__none__A": ("logreg", "A"), "ebm__none__B": ("ebm", "B")}
PER_CELL, N_BACKGROUND, N_DICE, N_CFS, THRESHOLD = 17, 100, 20, 3, 0.03
IMMUTABLE = ["age", "sex", "height_cm", "race_eth", "family_history_diabetes", "smoker"]
HISTORY = ["gdm_history", "baby_9lb", "htn_told", "sleep_trouble_told", "weight_change_1y_kg",
           "weight_change_10y_kg", "weight_change_10y_kg_na", "weight_gain_since_25_kg",
           "weight_gain_since_25_kg_na"]


def sample_patients(df: pd.DataFrame, seed: int) -> np.ndarray:
    """PER_CELL patients per race/ethnicity x sex cell, at random within cell."""
    rng = np.random.default_rng(seed)
    cells = df.groupby(["race_eth", "sex"]).indices
    return np.concatenate([rng.choice(cells[k], PER_CELL, replace=False) for k in sorted(cells)])


def fit_models(cfg, df, run):
    """Refit the 15 outer-fold and 3 LOCO models exactly as in run_dev.py; check against the
    committed per-fold results and out-of-fold predictions."""
    model, set_name = RUNS[run]
    feats, cat = feature_set(cfg, set_name)
    cv, seed = cfg["cv"], cfg["seed"]
    fold_models, loco_models = {}, {}
    folds, oof = nested_cv(
        df[feats], df[MAIN_LABEL], model, "none", cat, cv["outer_folds"], cv["inner_folds"],
        cv["n_iter"], cv["scoring"], cfg["preprocessing"]["knn_neighbors"],
        cfg["features"]["interactions"], seed, cv["outer_repeats"],
        on_fold=lambda i, s, tr, te: fold_models.__setitem__(i, (s.best_estimator_, tr, te)))
    loco_res = loco(df, feats, cat, MAIN_LABEL, model, "none", cfg, cv["n_iter"],
                    on_fold=lambda c, s, tr, te: loco_models.__setitem__(
                        c, (s.best_estimator_, tr.index.to_numpy(), te.index.to_numpy())))
    committed = pd.read_csv(Path("results/dev_cv_v2") / f"{run}.csv")
    check = folds[["repeat", "fold", "pr_auc"]].merge(
        committed[["repeat", "fold", "pr_auc"]], on=["repeat", "fold"], suffixes=("_refit", "_committed"))
    check.insert(0, "kind", "outer_fold")
    loco_ref = Path("results/dev_cv_v2/loco") / f"{run}.csv"
    lc = loco_res[["held_out_cycle", "pr_auc"]].rename(columns={"pr_auc": "pr_auc_refit"})
    lc["pr_auc_committed"] = (lc.held_out_cycle.map(pd.read_csv(loco_ref)
                                                    .set_index("held_out_cycle").pr_auc)
                              if loco_ref.exists() else np.nan)
    check = pd.concat([check, lc.assign(kind="loco")], ignore_index=True).assign(run=run)
    check["abs_diff"] = (check.pr_auc_refit - check.pr_auc_committed).abs()
    _, committed_oof = load_dev_oof(cfg, run)
    check.attrs["oof_max_abs_diff"] = float(np.abs(oof - committed_oof).max())
    return fold_models, loco_models, check


def explain_run(cfg, df, run, patients, fold_models, loco_models):
    model, set_name = RUNS[run]
    feats, cat = feature_set(cfg, set_name)
    seed, X = cfg["seed"], df[feats]
    codec = Codec(X, cat)
    models = [(f"fold{i}", *fold_models[i][:2]) for i in sorted(fold_models)]
    models += [(f"loco_{c}", *loco_models[c][:2]) for c in sorted(loco_models)]
    phis, lime_rows = {}, {}
    for name, pipe, tr in models:
        t0 = time.perf_counter()
        train = imputed_raw(pipe, X.iloc[tr])
        bg = codec.encode(train.iloc[np.random.default_rng(seed).choice(len(train), N_BACKGROUND,
                                                                        replace=False)])
        phis[name] = shap_values(pipe, codec, codec.encode(imputed_raw(pipe, X.iloc[patients])),
                                 bg, seed)
        if name in {f"fold{i}" for i in range(cfg["cv"]["outer_folds"])}:   # repeat 0: held out
            held = [j for j, pt in enumerate(patients) if pt in set(fold_models[int(name[4:])][2])]
            lime_rows[name] = (held, lime_weights(
                pipe, codec, codec.encode(train),
                codec.encode(imputed_raw(pipe, X.iloc[patients[held]])), seed))
        print(f"  {run} {name}: SHAP{' + LIME' if name in lime_rows else ''} "
              f"{time.perf_counter() - t0:.0f}s", flush=True)

    # held-out SHAP vs LIME, each patient by the repeat-0 model that had them in its test fold
    shap_h, lime_h = np.zeros((len(patients), len(feats))), np.zeros((len(patients), len(feats)))
    for name, (held, w) in lime_rows.items():
        shap_h[held], lime_h[held] = phis[name][held], w
    agree = agreement(shap_h, lime_h).assign(
        run=run, patient=patients, race_eth=df.race_eth.iloc[patients].map(RACE_ETH).to_numpy(),
        sex=df.sex.iloc[patients].to_numpy())
    rows = []
    for var, groups in [("overall", None), ("race_eth", agree.race_eth), ("sex", agree.sex)]:
        for g, part in ([("all", agree)] if groups is None else agree.groupby(groups)):
            o, s = bootstrap_mean(part.top5_overlap, seed=seed), bootstrap_mean(part.spearman, seed=seed)
            rows.append({"run": run, "variable": var, "group": g, "n": len(part),
                         "top5_overlap": o[0], "top5_overlap_ci_low": o[1], "top5_overlap_ci_high": o[2],
                         "spearman": s[0], "spearman_ci_low": s[1], "spearman_ci_high": s[2],
                         "spearman_n_undefined": int(part.spearman.isna().sum())})
    stab_rows, ranks = [], []
    for kind, prefix in [("outer_folds", "fold"), ("loco", "loco_")]:
        names = [n for n in phis if n.startswith(prefix)]
        summary, table = stability(np.stack([phis[n] for n in names]), feats, seed=seed)
        stab_rows.append({"run": run, "models": kind, **summary})
        ranks.append(table.assign(run=run, models=kind))
    np.savez_compressed(OUT / f"shap_{run}.npz", patients=patients, features=np.array(feats),
                        **phis)
    agree.to_csv(OUT / f"agreement_patients_{run}.csv", index=False, float_format="%.4f")
    return pd.DataFrame(rows), pd.DataFrame(stab_rows), pd.concat(ranks)


def heldout_p(df, feats, fold_models, outer_folds):
    p = np.full(len(df), np.nan)
    for i in range(outer_folds):
        pipe, _, te = fold_models[i]
        p[te] = pipe.predict_proba(df[feats].iloc[te])[:, 1]
    return p


def run_dice(cfg, df):
    seed, k = cfg["seed"], cfg["cv"]["outer_folds"]
    fitted = {run: joblib.load(CACHE / f"{run}.joblib") for run in RUNS}
    p0 = {run: heldout_p(df, feature_set(cfg, RUNS[run][1])[0], fitted[run]["fold"], k)
          for run in RUNS}
    eligible = np.flatnonzero(np.all([p0[r] >= THRESHOLD for r in RUNS], axis=0))
    chosen = np.sort(np.random.default_rng(seed).choice(eligible, N_DICE, replace=False))
    print(f"DiCE: {len(eligible)} patients predicted >= {THRESHOLD:.0%} by both models; "
          f"{N_DICE} sampled", flush=True)
    cf_rows = []
    for run in RUNS:
        feats, cat = feature_set(cfg, RUNS[run][1])
        numeric = [f for f in feats if f not in cat]
        vary = [f for f in feats if f not in IMMUTABLE]
        history = [f for f in HISTORY if f in feats]
        for j in chosen:
            i = next(i for i in range(k) if j in set(fitted[run]["fold"][i][2]))
            pipe, tr, _ = fitted[run]["fold"][i]
            raw = df[feats].iloc[tr]
            train = imputed_raw(pipe, raw)
            precision = {f: 0 for f in numeric
                         if np.all(np.mod(raw[f].dropna().to_numpy(dtype=float), 1) == 0)}
            data = dice_ml.Data(dataframe=train.assign(**{MAIN_LABEL: df[MAIN_LABEL].iloc[tr].to_numpy()}),
                                continuous_features=numeric, outcome_name=MAIN_LABEL,
                                continuous_features_precision=precision)
            dice = dice_ml.Dice(data, dice_ml.Model(model=pipe, backend="sklearn",
                                                    model_type="classifier"), method="random")
            query = imputed_raw(pipe, df[feats].iloc[[j]]).reset_index(drop=True)
            base = {"run": run, "patient": int(j), "fold": i, "p_query": float(pipe.predict_proba(query)[0, 1]),
                    "race_eth": RACE_ETH.get(df.race_eth.iloc[j]), "sex": df.sex.iloc[j], "note": NOTE}
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = dice.generate_counterfactuals(
                        query, total_CFs=N_CFS, desired_class=0, features_to_vary=vary,
                        stopping_threshold=THRESHOLD, random_seed=seed, verbose=False)
                cfs = res.cf_examples_list[0].final_cfs_df
            except Exception as e:                       # DiCE raises when it finds none
                cfs, base["error"] = None, f"{type(e).__name__}: {e}"[:200]
            if cfs is None or len(cfs) == 0:
                cf_rows.append({**base, "found": 0})
                continue
            cfs = cfs[feats].copy()
            cfs[numeric] = cfs[numeric].astype(float)
            m = cf_metrics(query.iloc[0], cfs, pipe.predict_proba(cfs)[:, 1], train, numeric,
                           history, THRESHOLD)
            for (_, c), (_, r) in zip(cfs.iterrows(), m.iterrows(), strict=True):
                diff = {f"cf_{f}": c[f] for f in r["changed"].split(";") if f}
                orig = {f"orig_{f}": query.iloc[0][f] for f in r["changed"].split(";") if f}
                cf_rows.append({**base, "found": len(cfs), **r.to_dict(), **orig, **diff})
        print(f"  DiCE {run}: done", flush=True)
    cf = pd.DataFrame(cf_rows)
    cf.to_csv(OUT / "dice_counterfactuals.csv", index=False, float_format="%.4f")
    summ = []
    for run, g in cf.groupby("run", sort=False):
        got = g[g.found > 0]
        summ.append({"run": run, "patients": g.patient.nunique(), "cfs_requested": N_DICE * N_CFS,
                     "cfs_found": len(got), "patients_with_none": int((g.found == 0).sum()),
                     "validity_of_requested": float(got.valid.sum() / (N_DICE * N_CFS)),
                     "validity_of_found": float(got.valid.mean()) if len(got) else np.nan,
                     "proximity_mean_mad": float(got.proximity.mean()),
                     "sparsity_mean": float(got.sparsity.mean()),
                     "plausible_share": float(got.plausible.mean()) if len(got) else np.nan,
                     "in_range_share": float(got.in_range.mean()) if len(got) else np.nan,
                     "no_history_change_share": float(got.no_history_change.mean()) if len(got) else np.nan,
                     "note": NOTE})
    pd.DataFrame(summ).to_csv(OUT / "dice_summary.csv", index=False, float_format="%.4f")
    return pd.DataFrame(summ)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="fit,shap,dice")
    ap.add_argument("--runs", default=",".join(RUNS))
    a = ap.parse_args()
    stages, runs = a.stage.split(","), a.runs.split(",")
    cfg = yaml.safe_load(open("configs/config.yaml"))
    df = load_dev(cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    patients = sample_patients(df, cfg["seed"])
    for run in runs:
        if "fit" in stages:
            t0 = time.perf_counter()
            fold_models, loco_models, check = fit_models(cfg, df, run)
            joblib.dump({"fold": fold_models, "loco": loco_models}, CACHE / f"{run}.joblib")
            check.to_csv(OUT / f"refit_check_{run}.csv", index=False, float_format="%.6f")
            print(f"{run}: refit {time.perf_counter() - t0:.0f}s; max |PR-AUC diff| vs committed "
                  f"{check.abs_diff.max():.2e}; max |OOF diff| {check.attrs['oof_max_abs_diff']:.2e}",
                  flush=True)
        if "shap" in stages:
            fitted = joblib.load(CACHE / f"{run}.joblib")
            agree, stab, ranks = explain_run(cfg, df, run, patients, fitted["fold"], fitted["loco"])
            agree.to_csv(OUT / f"agreement_summary_{run}.csv", index=False, float_format="%.4f")
            stab.to_csv(OUT / f"stability_summary_{run}.csv", index=False, float_format="%.4f")
            ranks.to_csv(OUT / f"stability_ranks_{run}.csv", index=False, float_format="%.4f")
            print(agree.round(3).to_string(index=False), stab.round(3).to_string(index=False),
                  sep="\n", flush=True)
    if "dice" in stages:
        print(run_dice(cfg, df).round(3).to_string(index=False))
    (OUT / "README.md").write_text(
        "Explanation analysis, development data 2011-2016 only (docs/analysis_plan.md, deviation "
        "2026-11-02 B). Exploratory.\n\nCounterfactuals (dice_*.csv) explain the model: they show "
        "which input changes would move the model's predicted risk below 3 %. They are not "
        "clinical advice, not causal claims and not recommendations for any person.\n\n"
        "DiCE implementation detail: numeric inputs whose observed training values are all whole "
        "numbers (flags, counts) are given 0 decimal places, so counterfactual values stay whole.\n"
        + json.dumps({"patients_per_cell": PER_CELL, "background": N_BACKGROUND,
                      "dice_patients": N_DICE, "cfs_per_patient": N_CFS, "threshold": THRESHOLD,
                      "immutable": IMMUTABLE}, indent=2) + "\n")
