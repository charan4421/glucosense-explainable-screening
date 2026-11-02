"""The single 2017-2018 temporal test (docs/analysis_plan.md section 9 and deviation "test protocol").

    python scripts/run_test.py --stage fit      # 2011-2016 only: refit, hash, write the run log
    python scripts/run_test.py --stage score    # load 2017-2018 and score ONCE

`fit` never loads the test file (it hashes its bytes). `score` refuses to run unless the tree is
clean, HEAD is the commit tagged `pre-test`, every hash matches the run log, and the output
folder does not exist yet. Nothing is rerun after the results are seen.

Primary: logistic regression / set A / one-hot, uncalibrated. Secondary (labelled): survey-
weighted ROC/PR-AUC, the intercept-recalibrated primary model, EBM / set B, the ADA-style score,
and subgroup tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so `src` imports work

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import yaml  # noqa: E402

from scripts.dev_dca import THRESHOLDS, dca_table  # noqa: E402
from scripts.run_dev import SUMMARY, load_dev  # noqa: E402
from src.data.nhanes import CYCLES, MAIN_LABEL, TEST_PATH  # noqa: E402
from src.evaluation.checkpoint import combo_hash  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    _logit,
    bootstrap_ci,
    calibration_stats,
    plot_calibration,
    summarize,
)
from src.evaluation.nested_cv import nested_cv  # noqa: E402
from src.evaluation.subgroups import RACE_ETH, subgroup_table  # noqa: E402
from src.evaluation.temporal import feature_set, fit_final  # noqa: E402
from src.models.models import categorical_encoding  # noqa: E402

# name -> (model, strategy, set, role)
RUNS = {
    "logreg__none__A": ("logreg", "none", "A", "primary"),
    "ebm__none__B": ("ebm", "none", "B", "secondary"),
    "ada_style__none__A": ("ada_style", "none", "A", "secondary (comparator)"),
}
PRIMARY = "logreg__none__A"
TAG = "pre-test"
MODELS = Path("reports/test_models")
RUN_LOG = MODELS / "run_log.json"
OUT = Path("results/test_2017_2018")
N_BOOT = 1000
CODE_GLOBS = ["src/**/*.py", "scripts/*.py", "configs/config.yaml"]

sklearn.set_config(working_memory=128)       # as in the development run (memory only)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted({p for g in CODE_GLOBS for p in Path(".").glob(g)}):
        h.update(str(f.as_posix()).encode() + b"\0" + f.read_bytes())
    return h.hexdigest()


def input_hashes(cfg: dict) -> dict:
    letter = CYCLES[cfg["data"]["temporal_test_cycle"]]
    raw = sorted(Path(cfg["data"]["raw_dir"]).glob(f"*_{letter}.XPT"))
    return {"config": sha256(Path("configs/config.yaml")), "code": code_hash(),
            "dev_data": sha256(Path(cfg["data"]["dev_processed_path"])),
            "test_data": sha256(Path(TEST_PATH)),
            "test_raw": {f.name: sha256(f) for f in raw}}


def git_state() -> dict:
    return {"commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"),
            "tag_commit": git("rev-list", "-n", "1", TAG), "dirty": git("status", "--porcelain")}


def p1(est, X) -> np.ndarray:
    return est.predict_proba(X)[:, 1]


def recalibrate(y, p) -> tuple[np.ndarray, float]:
    """Intercept re-estimated on (y, p) with the slope fixed at 1 (logit p as offset)."""
    a = calibration_stats(y, p)["intercept_offset"]
    return 1 / (1 + np.exp(-(a + _logit(p)))), a


CAL_KEYS = {"calibration_slope": "slope", "calibration_intercept": "intercept",
            "calibration_in_the_large": "intercept_offset",
            "observed_expected": "citl_ratio_observed_expected"}


def scores(y, p) -> dict:
    s, c = summarize(y, p), calibration_stats(y, p)
    return {"pr_auc": s["pr_auc"], "roc_auc": s["roc_auc"], "brier": s["brier"],
            "mean_predicted": c["citl_mean_predicted"],
            **{k: c[v] for k, v in CAL_KEYS.items()}}


def metric_table(y, p, n_boot: int = N_BOOT, seed: int = 42, recal: bool = False,
                 alpha: float = 0.05) -> dict:
    """Point estimates and percentile bootstrap 95 % intervals (rows resampled, one-class
    resamples skipped). With recal=True the predictions are intercept-recalibrated on (y, p), and
    the intercept is re-estimated inside every resample."""
    y, p = np.asarray(y), np.asarray(p)
    q = recalibrate(y, p)[0] if recal else p
    out = {"n": len(y), "n_pos": int(y.sum()), "prevalence": float(y.mean()), **scores(y, q)}
    if recal:
        out["recalibration_intercept"] = recalibrate(y, p)[1]
    rng = np.random.default_rng(seed)
    draws, skipped = [], 0
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            skipped += 1
            continue
        qb = recalibrate(y[idx], p[idx])[0] if recal else p[idx]
        draws.append(scores(y[idx], qb))
    d = pd.DataFrame(draws)
    for k in d:
        out[f"{k}_ci_low"], out[f"{k}_ci_high"] = np.percentile(d[k], [100 * alpha / 2,
                                                                       100 * (1 - alpha / 2)])
    out["n_boot"], out["n_boot_skipped"] = n_boot, skipped
    return out


def weighted_metrics(y, p, w, n_boot: int = N_BOOT, seed: int = 42) -> dict:
    s, ci = summarize(y, p, sample_weight=w), bootstrap_ci(y, p, n_boot, seed, sample_weight=w)
    return {"weighted_prevalence": s["prevalence"], "pr_auc": s["pr_auc"],
            "pr_auc_ci_low": ci["pr_auc_ci_low"], "pr_auc_ci_high": ci["pr_auc_ci_high"],
            "roc_auc": s["roc_auc"], "roc_auc_ci_low": ci["roc_auc_ci_low"],
            "roc_auc_ci_high": ci["roc_auc_ci_high"], "n_boot": n_boot,
            "n_boot_skipped": ci["n_skipped"]}


def stage_fit(cfg: dict) -> None:
    state = git_state()
    if state["dirty"] or state["commit"] != state["tag_commit"]:
        raise SystemExit(f"tree must be clean and HEAD tagged {TAG!r}: {state}")
    if not Path(TEST_PATH).exists():
        raise SystemExit(f"{TEST_PATH} missing: run python -m src.data.nhanes --test first")
    dev = load_dev(cfg)                      # asserts the test cycle is absent
    summary = pd.read_csv(SUMMARY)
    MODELS.mkdir(parents=True, exist_ok=True)
    log = {"git": state, "hashes": input_hashes(cfg), "n_dev": len(dev),
           "dev_positives": int(dev[MAIN_LABEL].sum()), "models": {}}
    for run, (model, strategy, set_name, role) in RUNS.items():
        feats, cat = feature_set(cfg, set_name)
        h = combo_hash(cfg, feats, cat, cfg["cv"]["n_iter"], cfg["data"]["dev_processed_path"],
                       categorical_encoding(model))
        est = fit_final(dev, feats, cat, MAIN_LABEL, model, strategy, cfg)
        path = MODELS / f"{run}.joblib"
        joblib.dump(est, path)
        p_dev = p1(est, dev[feats])
        rec = {"role": role, "set": set_name, "n_features": len(feats),
               "encoding": categorical_encoding(model), "settings_hash": h,
               "best_params": str(getattr(est, "best_params_", {})),
               "model_sha256": sha256(path),
               "dev_prediction_sha256": hashlib.sha256(np.round(p_dev, 10).tobytes()).hexdigest(),
               "dev_apparent": {k: float(v) for k, v in summarize(dev[MAIN_LABEL], p_dev).items()}}
        row = summary[(summary.model == model) & (summary.strategy == strategy)
                      & (summary.feature_set == set_name)]
        if len(row):                         # settings must be the ones the dev run used
            if row.run_hash.iloc[0] != h:
                raise SystemExit(f"{run}: settings hash {h} != development {row.run_hash.iloc[0]}")
            rec["dev_nested_cv"] = {k: float(row[k].iloc[0]) for k in
                                    ["pr_auc_mean", "pr_auc_sd", "roc_auc_mean", "roc_auc_sd",
                                     "brier_mean", "calibration_slope_mean",
                                     "calibration_intercept_mean",
                                     "citl_ratio_observed_expected_mean"]}
        else:                                # ADA-style: development reference, same folds
            cv = cfg["cv"]
            folds, _ = nested_cv(dev[feats], dev[MAIN_LABEL], model, strategy, cat,
                                 cv["outer_folds"], cv["inner_folds"], cv["n_iter"],
                                 cv["scoring"], cfg["preprocessing"]["knn_neighbors"],
                                 cfg["features"]["interactions"], cfg["seed"],
                                 cv.get("outer_repeats", 1))
            folds.to_csv(MODELS / f"{run}_dev_folds.csv", index=False)
            rec["dev_nested_cv"] = {
                "pr_auc_mean": folds.pr_auc.mean(), "pr_auc_sd": folds.pr_auc.std(),
                "roc_auc_mean": folds.roc_auc.mean(), "roc_auc_sd": folds.roc_auc.std(),
                "brier_mean": folds.brier.mean(),
                "calibration_slope_mean": folds.calibration_slope.mean(),
                "calibration_intercept_mean": folds.calibration_intercept.mean(),
                "citl_ratio_observed_expected_mean": folds.citl_ratio_observed_expected.mean()}
        log["models"][run] = rec
        print(f"{run}: fitted on 2011-2016 (n={len(dev)}); {rec['best_params']}; "
              f"model sha256 {rec['model_sha256'][:16]}", flush=True)
    RUN_LOG.write_text(json.dumps(log, indent=2, default=float))
    print(json.dumps(log, indent=2, default=float))


def check_before_score(cfg: dict) -> dict:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: the test has already been scored; no reruns")
    log = json.loads(RUN_LOG.read_text())
    state = git_state()
    problems = []
    if state["dirty"]:
        problems.append(f"working tree dirty:\n{state['dirty']}")
    if state["commit"] != state["tag_commit"] or state["commit"] != log["git"]["commit"]:
        problems.append(f"HEAD {state['commit']} vs tag {state['tag_commit']} vs run log "
                        f"{log['git']['commit']}")
    if input_hashes(cfg) != log["hashes"]:
        problems.append("input hashes differ from the run log")
    for run, rec in log["models"].items():
        if sha256(MODELS / f"{run}.joblib") != rec["model_sha256"]:
            problems.append(f"{run}: model file changed since the fit stage")
    if problems:
        raise SystemExit("refusing to score:\n" + "\n".join(problems))
    return log


def plot_dca(tab: pd.DataFrame, names: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = tab.threshold * 100
    for name in names:
        line, = ax.plot(x, tab[name], marker="o", label=name.replace("__", " / "))
        ax.fill_between(x, tab[f"{name}_ci_low"], tab[f"{name}_ci_high"],
                        color=line.get_color(), alpha=0.12, lw=0)
    ax.plot(x, tab.treat_all, "k:", label="treat all")
    ax.plot(x, tab.treat_none, "k-", lw=1, label="treat none")
    top = max(float(tab[f"{n}_ci_high"].max()) for n in names)
    ax.set(xlabel="Threshold probability (%)", ylabel="Net benefit", ylim=(-0.005, top * 1.1),
           title="Decision curve, 2017-2018 temporal test (95 % bootstrap bands)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def stage_score(cfg: dict) -> None:
    log = check_before_score(cfg)
    test = pd.read_parquet(TEST_PATH).reset_index(drop=True)
    cycle = cfg["data"]["temporal_test_cycle"]
    assert set(test.cycle) == {cycle}, sorted(test.cycle.unique())
    y, w = test[MAIN_LABEL].to_numpy(), test.wtmec2yr.to_numpy()    # one cycle: not divided
    seed = cfg["seed"]
    OUT.mkdir(parents=True)

    preds = {}
    for run, (_, _, set_name, _) in RUNS.items():
        preds[run] = p1(joblib.load(MODELS / f"{run}.joblib"), test[feature_set(cfg, set_name)[0]])
    preds[f"{PRIMARY}__recalibrated"] = recalibrate(y, preds[PRIMARY])[0]

    rows = []
    for run, (*_, role) in RUNS.items():
        dev_ref = log["models"][run]["dev_nested_cv"]
        versions = [("uncalibrated", False)] + ([("intercept_recalibrated", True)]
                                                if run == PRIMARY else [])
        for version, recal in versions:
            rows.append({"run": run, "role": role if not recal else "secondary",
                         "version": version, **metric_table(y, preds[run], N_BOOT, seed, recal),
                         "dev_pr_auc_mean": dev_ref["pr_auc_mean"],
                         "dev_roc_auc_mean": dev_ref["roc_auc_mean"]})
    main = pd.DataFrame(rows)
    main.to_csv(OUT / "test_metrics.csv", index=False, float_format="%.5f")

    weighted = pd.DataFrame([{"run": run, "role": "secondary (survey-weighted)",
                              "weighting": "WTMEC2YR",
                              **weighted_metrics(y, preds[run], w, N_BOOT, seed)}
                             for run in RUNS])
    weighted.to_csv(OUT / "test_weighted.csv", index=False, float_format="%.5f")

    dca_names = [PRIMARY, f"{PRIMARY}__recalibrated", "ebm__none__B", "ada_style__none__A"]
    dca = dca_table(y, {k: preds[k] for k in dca_names}, THRESHOLDS, N_BOOT, seed)
    dca.to_csv(OUT / "test_dca.csv", index=False, float_format="%.5f")
    plot_dca(dca, dca_names, OUT / "test_dca.png")

    sub = subgroup_table(y, preds[PRIMARY], {"sex": test.sex,
                                             "race_eth": test.race_eth.map(RACE_ETH)},
                         n_boot=N_BOOT, seed=seed).assign(run=PRIMARY)
    sub.to_csv(OUT / f"test_subgroups_{PRIMARY}.csv", index=False, float_format="%.4f")
    plot_calibration(y, preds[PRIMARY], OUT / f"test_calibration_{PRIMARY}.png",
                     title="Logistic regression / A, 2017-2018 (uncalibrated)")

    pd.DataFrame({"y": y, "wtmec2yr": w, "sex": test.sex, "race_eth": test.race_eth,
                  **{f"p_{k}": v for k, v in preds.items()}}).to_csv(
        OUT / "test_predictions.csv", index=False, float_format="%.6f")
    ada_folds = MODELS / "ada_style__none__A_dev_folds.csv"
    if ada_folds.exists():
        pd.read_csv(ada_folds).to_csv(OUT / "dev_ada_style__none__A_folds.csv", index=False,
                                      float_format="%.5f")
    (OUT / "run_log.json").write_text(json.dumps(
        {**log, "score": {"git": git_state(), "n_test": len(test),
                          "n_boot": N_BOOT, "seed": seed}}, indent=2, default=float))

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        cols = ["run", "version", "n", "n_pos", "prevalence", "pr_auc", "pr_auc_ci_low",
                "pr_auc_ci_high", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "brier",
                "calibration_slope", "calibration_intercept", "calibration_in_the_large",
                "observed_expected", "observed_expected_ci_low", "observed_expected_ci_high"]
        print(main[cols].round(4).to_string(index=False))
        print(weighted.round(4).to_string(index=False))
        print(dca.round(5).to_string(index=False))
        print(sub.drop(columns="run").round(4).to_string(index=False))
    print(f"-> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--stage", required=True, choices=["fit", "score"])
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    stage_fit(cfg) if a.stage == "fit" else stage_score(cfg)
