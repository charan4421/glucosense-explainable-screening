"""NHANES loader: downloads CDC XPT files and builds the undiagnosed-diabetes dataset.

Label: no self-reported diagnosis (DIQ010 != 1) AND (HbA1c >= 6.5 OR fasting glucose >= 126).
Lab values build the label only. They are never returned as features.
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


def build_cycle(raw_dir: str | Path, cycle: str) -> pd.DataFrame:
    raw_dir, L = Path(raw_dir), CYCLES[cycle]
    demo = _read(raw_dir, "DEMO", L)[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3"]]
    bmx = _read(raw_dir, "BMX", L)[["SEQN", "BMXBMI", "BMXWAIST", "BMXHT"]]
    bpx = _read(raw_dir, "BPX", L)
    bpx["systolic_bp"] = bpx[[c for c in bpx if c.startswith("BPXSY")]].mean(axis=1)
    bpx["diastolic_bp"] = bpx[[c for c in bpx if c.startswith("BPXDI")]].mean(axis=1)
    bpx = bpx[["SEQN", "systolic_bp", "diastolic_bp"]]
    diq = _read(raw_dir, "DIQ", L)[["SEQN", "DIQ010"]]
    mcq = _read(raw_dir, "MCQ", L)[["SEQN", "MCQ300C"]]   # relative had diabetes
    smq = _read(raw_dir, "SMQ", L)[["SEQN", "SMQ020"]]    # smoked >=100 cigarettes
    ghb = _read(raw_dir, "GHB", L)[["SEQN", "LBXGH"]]
    glu = _read(raw_dir, "GLU", L)[["SEQN", "LBXGLU"]]

    df = demo
    for part in (bmx, bpx, diq, mcq, smq, ghb, glu):
        df = df.merge(part, on="SEQN", how="left")

    df = df[(df.RIDAGEYR >= 20) & (df.DIQ010 != 1)]       # adults, no known diabetes
    has_lab = df.LBXGH.notna() | df.LBXGLU.notna()
    df = df[has_lab].copy()
    df["undiagnosed_diabetes"] = ((df.LBXGH >= 6.5) | (df.LBXGLU >= 126)).astype(int)

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
        "undiagnosed_diabetes": df.undiagnosed_diabetes,
    })
    return out.replace({"nan": np.nan, "<NA>": np.nan})


def build(raw_dir: str | Path, cycles: list[str]) -> pd.DataFrame:
    return pd.concat([build_cycle(raw_dir, c) for c in cycles], ignore_index=True)


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
    print(df.shape, "prevalence:", round(df.undiagnosed_diabetes.mean(), 4))
