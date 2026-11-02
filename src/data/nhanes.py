"""NHANES loader: downloads CDC XPT files and builds the undiagnosed-diabetes dataset.

Analysis sample: adults (>= 20 y) with HbA1c measured, no self-reported diagnosis (DIQ010 not 1),
excluding refused/don't know on DIQ010 (7, 9), anyone taking insulin (DIQ050 = 1) or diabetes
pills (DIQ070 = 1), and women pregnant at the exam (RIDEXPRG = 1). Borderline (DIQ010 = 3) is kept.

Labels:
- undiagnosed_diabetes (main): HbA1c >= 6.5 %.
- undiagnosed_diabetes_a1c_or_fpg (sensitivity analysis only): HbA1c >= 6.5 % OR fasting plasma
  glucose >= 126 mg/dL, defined only for the fasting subsample (FPG measured, WTSAF2YR > 0);
  missing for everyone else. 2015-2016 FPG is back-adjusted to the earlier instrument first
  (FPG_BACKWARD, CDC GLU_I note).
Weights WTMEC2YR / WTSAF2YR are kept for prevalence estimates (pooled = weight / number of cycles).
Lab values and weights build labels and estimates only. They are never used as features.

`python -m src.data.nhanes --download` builds the DEVELOPMENT cycles only (data.dev_cycles) into
data.dev_processed_path; the temporal test cycle is never read by this build.
`python -m src.data.nhanes --test` builds the test cycle alone into data/processed/nhanes_test.parquet
with the same rules (docs/analysis_plan.md, deviation "test protocol").
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

from src.data.derive import RAW_COLUMNS, assert_no_structural_missing, derive

CYCLES = {
    "2011-2012": "G",
    "2013-2014": "H",
    "2015-2016": "I",
    "2017-2018": "J",
}
# CDC file stems per component; the cycle letter is appended.
BASE_FILES = ["DEMO", "BMX", "BPX", "DIQ", "MCQ", "SMQ", "GHB", "GLU"]
FILES = BASE_FILES + [s for s in RAW_COLUMNS if s not in BASE_FILES]
# CDC moved the files in 2024; the old /Nchs/Nhanes/{start}-{end}/{stem}_{letter}.XPT path now
# returns an HTML page with status 200, so a bad URL is only detectable from the content.
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{start}/DataFiles/{stem}_{letter}.xpt"
XPT_MAGIC = b"HEADER RECORD*******LIBRARY HEADER RECORD"
# Raw gate columns kept (prefixed gate_) so tests can check skip-pattern handling. Never features.
GATE_COLUMNS = ["RIAGENDR", "RIDAGEYR", "PAQ605", "PAQ620", "PAQ635", "PAQ650", "PAQ665",
                "SMQ020", "SMQ040", "ALQ101", "ALQ110", "ALQ120Q", "DBD895", "BPQ020", "RHQ131"]


def _url(cycle: str, stem: str) -> str:
    start, _ = cycle.split("-")
    return BASE.format(start=start, stem=stem, letter=CYCLES[cycle])


def _is_xpt(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(len(XPT_MAGIC)) == XPT_MAGIC


def download(raw_dir: str | Path, cycles: list[str]) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for cycle in cycles:
        for stem in FILES:
            out = raw_dir / f"{stem}_{CYCLES[cycle]}.XPT"
            if out.exists() and _is_xpt(out):
                continue
            url = _url(cycle, stem)
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            if not r.content.startswith(XPT_MAGIC):
                raise RuntimeError(f"{url} did not return a SAS XPT file (got {r.content[:40]!r})")
            out.write_bytes(r.content)
            print("downloaded", out.name, f"{len(r.content) / 1e6:.1f} MB")


def fix_xpt_zero(df: pd.DataFrame) -> pd.DataFrame:
    """SAS XPORT stores 0 as 5.4e-79 when read by pandas; restore exact zeros so that
    `== 0` gates (no drinks, no meals out, ...) work."""
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask(df[num].abs() < 1e-30, 0.0)
    return df


def _read(raw_dir: Path, stem: str, letter: str, cols: list[str] | None = None) -> pd.DataFrame:
    df = fix_xpt_zero(pd.read_sas(raw_dir / f"{stem}_{letter}.XPT", format="xport", encoding=None))
    if cols is not None:
        df = df[["SEQN", *[c for c in cols if c in df]]]
    return df.drop_duplicates("SEQN")


MAIN_LABEL = "undiagnosed_diabetes"
SENS_LABEL = "undiagnosed_diabetes_a1c_or_fpg"

# CDC GLU_I analytic note: 2015-2016 glucose was measured on a Roche Cobas C311 that reads ~2 %
# higher than the C501 used before. Backward equation to the C501 scale: C501 = a * C311 + b.
# Applied to the SENSITIVITY label only; the main label (HbA1c) is unchanged.
# GLU_J documentation: "The Cobas C311 analyzer was used in 2017-2018 and 2015-2016", so the same
# backward equation applies (section 9 FPG rule).
FPG_BACKWARD = {"2015-2016": (0.9776, 0.4994), "2017-2018": (0.9776, 0.4994)}

# Raw items that cannot be reproduced with the 2011-2016 definition in a cycle: left missing (the
# training-fitted imputers fill them). 2017-2018 redesigned the alcohol questionnaire (ALQ111/121/142).
UNMAPPED = {"2017-2018": ["ALQ101", "ALQ110", "ALQ120Q", "ALQ120U", "ALQ141Q", "ALQ141U"]}
TEST_PATH = "data/processed/nhanes_test.parquet"


def label_frame(df: pd.DataFrame, exclude_pregnant: bool = True,
                fpg_backward: tuple[float, float] | None = None) -> pd.DataFrame:
    """Apply the analysis-sample rules and add both labels to merged raw NHANES columns.

    Needs RIDAGEYR, DIQ010, DIQ050, DIQ070, LBXGH, LBXGLU, WTSAF2YR, and RIDEXPRG when
    exclude_pregnant (women pregnant at the exam are removed: HbA1c is lower in pregnancy).
    fpg_backward=(a, b) rescales FPG as a * FPG + b before the sensitivity label's >= 126 cut-off.
    """
    adult = df.RIDAGEYR >= 20
    no_dx = df.DIQ010.notna() & ~df.DIQ010.isin([1, 7, 9])     # keeps 2 (no) and 3 (borderline)
    on_meds = (df.DIQ050 == 1) | (df.DIQ070 == 1)               # insulin or diabetes pills
    keep = adult & no_dx & ~on_meds & df.LBXGH.notna()
    if exclude_pregnant:
        keep &= df.RIDEXPRG != 1                                # 1 = pregnant at exam
    df = df[keep].copy()
    df[MAIN_LABEL] = (df.LBXGH >= 6.5).astype(int)
    fasting = df.LBXGLU.notna() & (df.WTSAF2YR > 0)
    fpg = df.LBXGLU if fpg_backward is None else fpg_backward[0] * df.LBXGLU + fpg_backward[1]
    sens = ((df.LBXGH >= 6.5) | (fpg >= 126)).astype("Int64")
    df[SENS_LABEL] = sens.where(fasting, pd.NA).astype("Int64")
    return df


def build_cycle(raw_dir: str | Path, cycle: str, log: list | None = None) -> pd.DataFrame:
    raw_dir, L = Path(raw_dir), CYCLES[cycle]
    demo = _read(raw_dir, "DEMO", L, ["RIAGENDR", "RIDAGEYR", "RIDRETH3", "WTMEC2YR",
                                      *RAW_COLUMNS["DEMO"]])
    bmx = _read(raw_dir, "BMX", L, ["BMXBMI", "BMXWAIST", "BMXHT"])
    bpx = _read(raw_dir, "BPX", L)
    bpx["systolic_bp"] = bpx[[c for c in bpx if c.startswith("BPXSY")]].mean(axis=1)
    bpx["diastolic_bp"] = bpx[[c for c in bpx if c.startswith("BPXDI")]].mean(axis=1)
    bpx = bpx[["SEQN", "systolic_bp", "diastolic_bp"]]
    parts = [bmx, bpx,
             _read(raw_dir, "DIQ", L, ["DIQ010", "DIQ050", "DIQ070"]),
             _read(raw_dir, "MCQ", L, ["MCQ300C"]),             # relative had diabetes
             _read(raw_dir, "SMQ", L, ["SMQ020", *RAW_COLUMNS["SMQ"]]),
             _read(raw_dir, "GHB", L, ["LBXGH"]),
             _read(raw_dir, "GLU", L, ["LBXGLU", "WTSAF2YR"])]
    parts += [_read(raw_dir, stem, L, cols) for stem, cols in RAW_COLUMNS.items()
              if stem not in ("DEMO", "SMQ")]
    df = demo
    for part in parts:
        df = df.merge(part, on="SEQN", how="left")
    for col in UNMAPPED.get(cycle, []):
        if col in df:
            raise ValueError(f"{col} exists in {cycle}; revisit UNMAPPED")
        df[col] = np.nan

    adj = FPG_BACKWARD.get(cycle)
    n_with_pregnant = len(label_frame(df, exclude_pregnant=False))
    unadjusted = label_frame(df)
    df = label_frame(df, fpg_backward=adj)
    if log is not None:
        log.append({"cycle": cycle, "n": len(df), "pregnant_excluded": n_with_pregnant - len(df),
                    "main_positives": int(df[MAIN_LABEL].sum()),
                    "sens_positives_unadjusted": int(unadjusted[SENS_LABEL].sum()),
                    "sens_positives": int(df[SENS_LABEL].sum()),
                    "fasting_n": int(df[SENS_LABEL].notna().sum())})

    derived = derive(df)
    assert_no_structural_missing(df, derived)
    out = pd.DataFrame({
        "age": df.RIDAGEYR,
        "sex": df.RIAGENDR.map({1: "male", 2: "female"}),
        "bmi": df.BMXBMI,
        "waist_cm": df.BMXWAIST,
        "height_cm": df.BMXHT,
        "systolic_bp": df.systolic_bp,
        "diastolic_bp": df.diastolic_bp,
        "family_history_diabetes": df.MCQ300C.map({1.0: "yes", 2.0: "no"}),
        "smoker": df.SMQ020.map({1.0: "yes", 2.0: "no"}),
        "race_eth": df.RIDRETH3.astype("Int64").astype(str),
        "cycle": cycle,
        "wtmec2yr": df.WTMEC2YR,
        "wtsaf2yr": df.WTSAF2YR,
        MAIN_LABEL: df[MAIN_LABEL],
        SENS_LABEL: df[SENS_LABEL],
    })
    out = pd.concat([out, derived, df[GATE_COLUMNS].add_prefix("gate_")], axis=1)
    return out.replace({"nan": np.nan, "<NA>": np.nan}).reset_index(drop=True)


def build(raw_dir: str | Path, cycles: list[str], log: list | None = None) -> pd.DataFrame:
    return pd.concat([build_cycle(raw_dir, c, log) for c in cycles], ignore_index=True)


def prevalence_report(df: pd.DataFrame) -> pd.DataFrame:
    """Unweighted and survey-weighted prevalence per cycle and pooled.

    Main label weighted with WTMEC2YR; the sensitivity label (fasting subsample) with WTSAF2YR.
    Pooled weights are the 2-year weights divided by the number of cycles combined.
    """
    n_cycles = df.cycle.nunique()

    def row(g: pd.DataFrame, divisor: int) -> dict:
        w = g.wtmec2yr / divisor
        s = g[g[SENS_LABEL].notna()]
        ws = s.wtsaf2yr / divisor
        ys = s[SENS_LABEL].astype(int)
        return {
            "n": len(g), "positives": int(g[MAIN_LABEL].sum()),
            "unweighted": g[MAIN_LABEL].mean(),
            "weighted": (w * g[MAIN_LABEL]).sum() / w.sum(),
            "fasting_n": len(s),
            "sens_unweighted": ys.mean(),
            "sens_weighted": (ws * ys).sum() / ws.sum(),
        }

    rows = {c: row(g, 1) for c, g in df.groupby("cycle")}
    rows["pooled"] = row(df, n_cycles)
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--test", action="store_true", help="build the temporal test cycle alone")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    d = cfg["data"]
    if args.test:
        cycle = d["temporal_test_cycle"]
        if args.download:
            download(d["raw_dir"], [cycle])
        log: list = []
        df = build(d["raw_dir"], [cycle], log)
        assert set(df.cycle) == {cycle}
        df.to_parquet(TEST_PATH, index=False)
        # sample counts only: labels are not summarised before scoring
        print(f"{cycle}: n={len(df)}, pregnant excluded={log[0]['pregnant_excluded']} -> {TEST_PATH}")
        raise SystemExit(0)
    cycles = d["dev_cycles"]
    assert d["temporal_test_cycle"] not in cycles, "the test cycle must not be built here"
    if args.download:
        download(d["raw_dir"], cycles)
    log: list = []
    df = build(d["raw_dir"], cycles, log)
    Path(d["dev_processed_path"]).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d["dev_processed_path"], index=False)
    print("shape:", df.shape, "cycles:", sorted(df.cycle.unique()))
    print("\nsample and label changes per cycle:")
    print(pd.DataFrame(log).to_string(index=False))
    print("\nmissing % per column:")
    print((df.isna().mean() * 100).round(1).to_string())
    print(f"\nprevalence ({MAIN_LABEL} = HbA1c >= 6.5; sens = HbA1c or FPG, fasting subsample):")
    with pd.option_context("display.width", 160):
        print(prevalence_report(df).astype(float).round(4).to_string())
