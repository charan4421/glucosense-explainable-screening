"""Resume logic: a combination is skipped only when its file exists and the hash matches."""
import pandas as pd

from src.evaluation.checkpoint import build_summary, combo_path, is_done, run_hash, save_combo

CFG = {"seed": 42, "cv": {"outer_folds": 5, "inner_folds": 3}, "models": ["xgboost"]}


def folds(model="xgboost", strategy="none", n_iter=5):
    return pd.DataFrame({"model": model, "strategy": strategy, "fold": range(3), "n_iter": n_iter,
                         "pr_auc": [0.1, 0.2, 0.3], "roc_auc": 0.7, "brier": 0.03,
                         "calibration_slope": 1.0, "calibration_intercept": 0.0,
                         "citl_difference": 0.0, "citl_ratio_observed_expected": 1.0,
                         "n_pos": 30, "n": 1000, "fit_seconds": 1.0})


def test_hash_is_stable_and_sensitive():
    assert run_hash(CFG, 42, 5) == run_hash(dict(reversed(list(CFG.items()))), 42, 5)
    assert run_hash(CFG, 42, 5) != run_hash(CFG, 42, 15)
    assert run_hash(CFG, 42, 5) != run_hash(CFG, 0, 5)
    assert run_hash(CFG, 42, 5) != run_hash({**CFG, "cv": {"outer_folds": 10, "inner_folds": 3}}, 42, 5)


def test_skip_when_hash_matches_rerun_when_it_differs(tmp_path):
    path = combo_path(tmp_path, "xgboost", "none")
    h5 = run_hash(CFG, 42, 5)
    assert not is_done(path, h5)                        # no file yet -> run
    save_combo(folds(), path, h5)
    assert is_done(path, h5)                            # same config/seed/n_iter -> skip
    assert not is_done(path, run_hash(CFG, 42, 15))     # n_iter changed -> rerun
    assert not is_done(path, run_hash({**CFG, "seed": 1}, 1, 5))   # config/seed changed -> rerun


def test_empty_or_corrupt_file_reruns(tmp_path):
    path = combo_path(tmp_path, "xgboost", "none")
    path.write_text("")
    assert not is_done(path, run_hash(CFG, 42, 5))


def test_summary_mean_and_sd(tmp_path):
    h = run_hash(CFG, 42, 5)
    save_combo(folds("xgboost", "none"), combo_path(tmp_path, "xgboost", "none"), h)
    save_combo(folds("mlp", "none"), combo_path(tmp_path, "mlp", "none"), h)
    s = build_summary(tmp_path, tmp_path / "summary.csv").set_index(["model", "strategy"])
    assert len(s) == 2
    assert abs(s.loc[("xgboost", "none"), "pr_auc_mean"] - 0.2) < 1e-9
    assert abs(s.loc[("xgboost", "none"), "pr_auc_sd"] - 0.1) < 1e-9
    assert s.loc[("xgboost", "none"), "n_folds"] == 3
