"""NHANES loader: downloads CDC XPT files and builds the undiagnosed-diabetes dataset.

Analysis sample: adults (>= 20 y) with HbA1c measured, no self-reported diagnosis (DIQ010 not 1),
excluding refused/don't know on DIQ010 (7, 9) and anyone taking insulin (DIQ050 = 1) or diabetes
pills (DIQ070 = 1). Borderline (DIQ010 = 3) is kept.

Labels:
- undiagnosed_diabetes (main): HbA1c >= 6.5 %.
- undiagnosed_diabetes_a1c_or_fpg (sensitivity analysis only): HbA1c >= 6.5 % OR fasting plasma
  glucose >= 126 mg/dL, defined only for the fasting subsample (FPG measured, WTSAF2YR > 0);
  missing for everyone else.
Weights WTMEC2YR / WTSAF2YR are kept for prevalence estimates (pooled = weight / number of cycles).
Lab values and weights build labels and estimates only. They are never used as features.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

CYCLES = {
    "2011-2012": "G",
    "2013-2014": "H",
    "2015-2016": "I",
    "2017-2018": "J",
}
# CDC file stems per component; the cycle letter is appended.
FILES = ["DEMO", "BMX", "BPX", "DIQ", "MCQ", "SMQ", "GHB", "GLU"]
# CDC moved the files in 2024; the old /Nchs/Nhanes/{start}-{end}/{stem}_{letter}.XPT path now
# returns an HTML page with status 200, so a bad URL is only detectable from the content.
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{start}/DataFiles/{stem}_{letter}.xpt"
XPT_MAGIC = b"HEADER RECORD*******LIBRARY HEADER RECORD"


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


def _read(raw_dir: Path, stem: str, letter: str) -> pd.DataFrame:
    return pd.read_sas(raw_dir / f"{stem}_{letter}.XPT", format="xport")


MAIN_LABEL = "undiagnosed_diabetes"
SENS_LABEL = "undiagnosed_diabetes_a1c_or_fpg"


def label_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the analysis-sample rules and add both labels to merged raw NHANES columns.

    Needs RIDAGEYR, DIQ010, DIQ050, DIQ070, LBXGH, LBXGLU, WTSAF2YR.
    """
    adult = df.RIDAGEYR >= 20
    no_dx = df.DIQ010.notna() & ~df.DIQ010.isin([1, 7, 9])     # keeps 2 (no) and 3 (borderline)
    on_meds = (df.DIQ050 == 1) | (df.DIQ070 == 1)               # insulin or diabetes pills
    df = df[adult & no_dx & ~on_meds & df.LBXGH.notna()].copy()
    df[MAIN_LABEL] = (df.LBXGH >= 6.5).astype(int)
    fasting = df.LBXGLU.notna() & (df.WTSAF2YR > 0)
    sens = ((df.LBXGH >= 6.5) | (df.LBXGLU >= 126)).astype("Int64")
    df[SENS_LABEL] = sens.where(fasting, pd.NA).astype("Int64")
    return df


def build_cycle(raw_dir: str | Path, cycle: str) -> pd.DataFrame:
    raw_dir, L = Path(raw_dir), CYCLES[cycle]
    demo = _read(raw_dir, "DEMO", L)[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "WTMEC2YR"]]
    bmx = _read(raw_dir, "BMX", L)[["SEQN", "BMXBMI", "BMXWAIST", "BMXHT"]]
    bpx = _read(raw_dir, "BPX", L)
    bpx["systolic_bp"] = bpx[[c for c in bpx if c.startswith("BPXSY")]].mean(axis=1)
    bpx["diastolic_bp"] = bpx[[c for c in bpx if c.startswith("BPXDI")]].mean(axis=1)
    bpx = bpx[["SEQN", "systolic_bp", "diastolic_bp"]]
    diq = _read(raw_dir, "DIQ", L)[["SEQN", "DIQ010", "DIQ050", "DIQ070"]]
    mcq = _read(raw_dir, "MCQ", L)[["SEQN", "MCQ300C"]]   # relative had diabetes
    smq = _read(raw_dir, "SMQ", L)[["SEQN", "SMQ020"]]    # smoked >=100 cigarettes
    ghb = _read(raw_dir, "GHB", L)[["SEQN", "LBXGH"]]
    glu = _read(raw_dir, "GLU", L)[["SEQN", "LBXGLU", "WTSAF2YR"]]

    df = demo
    for part in (bmx, bpx, diq, mcq, smq, ghb, glu):
        df = df.merge(part, on="SEQN", how="left")
    df = label_frame(df)

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
    return out.replace({"nan": np.nan, "<NA>": np.nan})


def build(raw_dir: str | Path, cycles: list[str]) -> pd.DataFrame:
    return pd.concat([build_cycle(raw_dir, c) for c in cycles], ignore_index=True)


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
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    d = cfg["data"]
    if args.download:
        download(d["raw_dir"], d["nhanes_cycles"])
    df = build(d["raw_dir"], d["nhanes_cycles"])
    Path(d["processed_path"]).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d["processed_path"], index=False)
    print("shape:", df.shape)
    print("\nmissing % per column:")
    print((df.isna().mean() * 100).round(1).to_string())
    print(f"\nprevalence ({MAIN_LABEL} = HbA1c >= 6.5; sens = HbA1c or FPG, fasting subsample):")
    with pd.option_context("display.width", 160):
        print(prevalence_report(df).astype(float).round(4).to_string())
