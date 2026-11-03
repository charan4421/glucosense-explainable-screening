"""Explanation figures for the README (docs/figures/), development data 2011-2016 only.
Exploratory; explanations describe the model, not cause and effect. 2017-2018 is never loaded.

  1. Global SHAP (bar + beeswarm) for logistic regression / A and EBM / B, from the committed
     held-out SHAP values (results/dev_cv_v2_explain/shap_*.npz: each of the 204 patients
     explained by the repeat-0 outer-fold model that had them in its test fold). Feature values
     for the colours come from that fold's own imputation (reports/explain_cache/).
  2. EBM's built-in global term importance, from the frozen EBM / B (refit on all of 2011-2016).
  3. The fictional sample patient (examples/sample_patient.json) under the frozen primary model:
     SHAP waterfall and LIME bars side by side (same settings as scripts/dev_explain.py:
     permutation SHAP, 100 background rows, LIME defaults, seed 42).
  4. DiCE for the same patient: 3 counterfactuals below 3 %, same settings and immutable inputs
     as scripts/dev_explain.py, with a realism check (training 1st-99th percentile).

    python scripts/dev_figures.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so `src` imports work

import dice_ml  # noqa: E402
import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import yaml  # noqa: E402

from scripts.demo import SAMPLE  # noqa: E402
from scripts.dev_explain import CACHE, IMMUTABLE, N_BACKGROUND, N_CFS, RUNS, THRESHOLD  # noqa: E402
from scripts.run_dev import load_dev  # noqa: E402
from scripts.run_test import MODELS  # noqa: E402
from src.data.nhanes import MAIN_LABEL  # noqa: E402
from src.evaluation.temporal import feature_set  # noqa: E402
from src.explain.analysis import Codec, imputed_raw, lime_weights, shap_values  # noqa: E402
from src.models.portable import INPUTS  # noqa: E402

FIG = Path("docs/figures")
EXPLAIN = Path("results/dev_cv_v2_explain")
TITLES = {"logreg__none__A": "Logistic regression / set A (primary)",
          "ebm__none__B": "EBM / set B (secondary)"}


def heldout(cfg, df, run):
    """Held-out SHAP values and imputed feature values for the 204 explained patients."""
    npz = np.load(EXPLAIN / f"shap_{run}.npz")
    patients, feats = npz["patients"], [str(f) for f in npz["features"]]
    fold = joblib.load(CACHE / f"{run}.joblib")["fold"]
    codec = Codec(df[feats], feature_set(cfg, RUNS[run][1])[1])
    phi, vals = np.zeros((len(patients), len(feats))), np.zeros((len(patients), len(feats)))
    for i in range(cfg["cv"]["outer_folds"]):                       # repeat 0
        pipe, _, te = fold[i]
        held = np.flatnonzero(np.isin(patients, te))
        phi[held] = npz[f"fold{i}"][held]
        vals[held] = codec.encode(imputed_raw(pipe, df[feats].iloc[patients[held]]))
    return phi, vals, feats


def global_plots(cfg, df, run, top=15):
    phi, vals, feats = heldout(cfg, df, run)
    exp = shap.Explanation(values=phi, data=vals, feature_names=feats)
    for kind, plot in [("bar", shap.plots.bar), ("beeswarm", shap.plots.beeswarm)]:
        plt.figure()
        plot(exp, max_display=top, show=False)
        plt.title(f"{TITLES[run]}: SHAP {kind}, held-out, 2011-2016", fontsize=10)
        plt.tight_layout()
        plt.savefig(FIG / f"shap_{kind}_{run}.png", dpi=130, bbox_inches="tight")
        plt.close("all")


def ebm_importance(top=15):
    pipe = joblib.load(MODELS / "ebm__none__B.joblib").best_estimator_
    ebm = pipe.named_steps["model"]
    imp = pd.Series(ebm.term_importances(), index=ebm.term_names_).sort_values().tail(top)
    names = dict(enumerate(pipe.named_steps["columns"].get_feature_names_out()))
    imp.index = [" & ".join(names.get(int(t[8:]), t) if t.startswith("feature_") else t
                            for t in s.split(" & ")) for s in imp.index]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values, color="#4C72B0")
    ax.set_xlabel("Mean |score| (EBM term importance, logit scale)")
    ax.set_title("EBM / set B: built-in term importance\n(frozen model, refit on 2011-2016)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "ebm_term_importance.png", dpi=130)
    plt.close(fig)
    return imp


def local_example(cfg, df, patient):
    feats, cat = feature_set(cfg, "A")
    pipe = joblib.load(MODELS / "logreg__none__A.joblib").best_estimator_
    seed, codec = cfg["seed"], Codec(df[feats], cat)
    train = imputed_raw(pipe, df[feats])
    bg = codec.encode(train.iloc[np.random.default_rng(seed).choice(len(train), N_BACKGROUND,
                                                                    replace=False)])
    x = codec.encode(patient[feats])
    phi = shap_values(pipe, codec, x, bg, seed)[0]
    lime = lime_weights(pipe, codec, codec.encode(train), x, seed)[0]
    p = float(pipe.predict_proba(patient[feats])[0, 1])
    base = float(codec.predictor(pipe)(bg).mean())
    shown = [f"{v}" for v in patient[feats].iloc[0]]

    plt.figure()
    shap.plots.waterfall(shap.Explanation(values=phi, base_values=base, data=np.array(shown),
                                          feature_names=feats), max_display=10, show=False)
    plt.title(f"SHAP: change in predicted risk (probability)\nfrom the average {base:.1%} "
              f"to {p:.1%}", fontsize=10)
    plt.savefig(FIG / "_shap_local.png", dpi=130, bbox_inches="tight")
    plt.close("all")

    order = np.argsort(np.abs(lime))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(np.array(feats)[order], lime[order],
            color=["#D55E00" if w > 0 else "#0072B2" for w in lime[order]])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("LIME weight (local linear surrogate, probability scale)")
    ax.set_title("LIME: same patient, same model", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "_lime_local.png", dpi=130)
    plt.close(fig)

    a, b = plt.imread(FIG / "_shap_local.png"), plt.imread(FIG / "_lime_local.png")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios":
                                                                  [a.shape[1] / a.shape[0],
                                                                   b.shape[1] / b.shape[0]]})
    for ax, img in zip(axes, [a, b], strict=True):
        ax.imshow(img)
        ax.axis("off")
    fig.suptitle("Fictional sample patient, frozen primary model: SHAP (left) vs LIME (right)")
    fig.tight_layout()
    fig.savefig(FIG / "local_shap_vs_lime_sample_patient.png", dpi=130)
    plt.close(fig)
    for f in ["_shap_local.png", "_lime_local.png"]:
        (FIG / f).unlink()
    return p, base, dict(zip(feats, phi, strict=True)), dict(zip(feats, lime, strict=True))


def dice_example(cfg, df, patient):
    feats, cat = feature_set(cfg, "A")
    numeric = [f for f in feats if f not in cat]
    pipe = joblib.load(MODELS / "logreg__none__A.joblib").best_estimator_
    train = imputed_raw(pipe, df[feats])
    data = dice_ml.Data(dataframe=train.assign(**{MAIN_LABEL: df[MAIN_LABEL].to_numpy()}),
                        continuous_features=numeric, outcome_name=MAIN_LABEL)
    dice = dice_ml.Dice(data, dice_ml.Model(model=pipe, backend="sklearn",
                                            model_type="classifier"), method="random")
    query = patient[feats].reset_index(drop=True)
    for c in numeric:
        query[c] = query[c].astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = dice.generate_counterfactuals(
            query, total_CFs=N_CFS, desired_class=0, stopping_threshold=THRESHOLD,
            features_to_vary=[f for f in feats if f not in IMMUTABLE], random_seed=cfg["seed"],
            verbose=False)
    cfs = res.cf_examples_list[0].final_cfs_df[feats].reset_index(drop=True)
    cfs[numeric] = cfs[numeric].astype(float)
    lo, hi = train[numeric].quantile(0.01), train[numeric].quantile(0.99)
    p_cf = pipe.predict_proba(cfs)[:, 1]
    table = pd.DataFrame({"input": feats, "original": query.iloc[0].to_numpy()})
    notes = []
    for k in range(len(cfs)):
        changed = [f for f in feats if str(cfs.at[k, f]) != str(query.at[0, f])
                   and not (f in numeric and np.isclose(float(cfs.at[k, f]), float(query.at[0, f])))]
        out = [f for f in changed if f in numeric and not lo[f] <= cfs.at[k, f] <= hi[f]]
        table[f"counterfactual {k + 1}"] = [
            (f"**{cfs.at[k, f]:.1f}**" if f in numeric else f"**{cfs.at[k, f]}**") if f in changed
            else "" for f in feats]
        notes.append({"cf": k + 1, "p": float(p_cf[k]), "changed": changed, "out_of_range": out,
                      "realistic_range": not out})
    return table, notes, float(pipe.predict_proba(query)[0, 1]), lo, hi


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/config.yaml"))
    df = load_dev(cfg)
    FIG.mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        global_plots(cfg, df, run)
    print("EBM top terms:\n", ebm_importance().iloc[::-1].round(3).to_string())
    patient = json.loads(SAMPLE.read_text())
    patient = pd.DataFrame([{c: patient[c] for c in INPUTS}]).astype({"race_eth": str})
    p, base, phi, lime = local_example(cfg, df, patient)
    print(f"\nsample patient: p={p:.4f} base={base:.4f}")
    print(pd.DataFrame({"shap": phi, "lime": lime}).round(4).to_string())
    table, notes, p0, lo, hi = dice_example(cfg, df, patient)
    table.to_csv(FIG / "dice_sample_patient.csv", index=False)
    print(f"\nDiCE (p={p0:.4f}):\n{table.to_string(index=False)}\n{json.dumps(notes, indent=1)}")
    print(pd.DataFrame({"p1": lo, "p99": hi}).round(1).to_string())
