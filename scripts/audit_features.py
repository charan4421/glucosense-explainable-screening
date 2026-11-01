"""Audit of candidate NON-LAB screening features in NHANES 2011-2016, plus a 2005-2016 pooling check.

No models, no 2017-2018 data. Downloads public questionnaire/exam XPT files and their CDC codebook
pages into data/raw (gitignored), then writes the tables to docs/feature_audit_tables.md.

- Meaning and target population come from the CDC codebook (SAS label, "Target").
- Missingness is measured on our development analysis sample (2011-2016 adults after the label
  exclusions, n = 13,355), only in the cycles where the variable exists.
- Refused / don't know codes are read from each variable's codebook value table.
- "Subset" gates are checked against the data: the share answering when the gate is true vs false.

    python scripts/audit_features.py
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so `src` imports work

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from src.data.nhanes import XPT_MAGIC, label_frame  # noqa: E402

RAW = Path("data/raw")
DOCS = RAW / "docs"
DEV = {"2011-2012": "G", "2013-2014": "H", "2015-2016": "I"}
EARLY = {"2005-2006": "D", "2007-2008": "E", "2009-2010": "F"}
ALL = {**EARLY, **DEV}
URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{start}/DataFiles/{stem}_{letter}.{ext}"

# (group, file stem, variable, flag). An empty flag = plausible feature; a flag = borderline, for
# the user to decide. Gates (for "asked of a subset") are in GATES below and verified on the data.
SPEC = [
    # physical activity (GPAQ-style, 2007+)
    *[("physical activity", "PAQ", v, "") for v in [
        "PAQ605", "PAQ610", "PAD615", "PAQ620", "PAQ625", "PAD630", "PAQ635", "PAQ640", "PAD645",
        "PAQ650", "PAQ655", "PAD660", "PAQ665", "PAQ670", "PAD675", "PAD680", "PAQ710", "PAQ715"]],
    # sleep
    *[("sleep", "SLQ", v, "") for v in [
        "SLD010H", "SLD012", "SLQ030", "SLQ040", "SLQ050", "SLQ060", "SLQ120"]],
    # smoking
    *[("smoking", "SMQ", v, "") for v in [
        "SMQ020", "SMQ040", "SMD030", "SMD641", "SMD650", "SMQ050Q", "SMQ050U"]],
    # alcohol
    *[("alcohol", "ALQ", v, "") for v in [
        "ALQ101", "ALQ110", "ALQ120Q", "ALQ120U", "ALQ130", "ALQ141Q", "ALQ141U", "ALQ151"]],
    # diet quality: behaviour questionnaire + day-1 24-h recall totals
    *[("diet quality", "DBQ", v, "") for v in [
        "DBQ700", "DBD895", "DBD900", "DBD905", "DBD910", "DBQ197"]],
    *[("diet quality", "DR1TOT", v, "single 24-h recall (day 1): noisy for one person; has its "
       "own dietary weight WTDRD1") for v in [
        "DR1TKCAL", "DR1TCARB", "DR1TSUGR", "DR1TFIBE", "DR1TTFAT", "DR1TSFAT", "DR1TSODI",
        "DR1TALCO", "DR1DRSTZ"]],
    # education / income / food security / acculturation / insurance / work
    *[("education, income, food security", "DEMO", v, "") for v in [
        "DMDEDUC2", "INDFMPIR", "INDHHIN2", "DMDMARTL", "DMDHHSIZ", "DMDBORN4", "DMDCITZN",
        "DMDYRSUS", "SIALANG"]],
    *[("education, income, food security", "FSQ", v, "") for v in ["FSDHH", "FSDAD", "FSQ165"]],
    ("education, income, food security", "HIQ", "HIQ011",
     "healthcare access: drives being diagnosed vs undiagnosed (selection into the label)"),
    *[("education, income, food security", "OCQ", v, "") for v in ["OCD150", "OCQ180"]],
    # weight history
    *[("weight history", "WHQ", v, "") for v in [
        "WHD010", "WHD020", "WHQ030", "WHQ040", "WHD050", "WHD110", "WHD120", "WHD130",
        "WHD140", "WHQ150", "WHQ070"]],
    ("weight history", "WHQ", "WHQ060",
     "unintentional weight loss is a symptom of hyperglycaemia (reverse causation)"),
    # blood pressure / hypertension
    ("blood pressure, hypertension", "BPQ", "BPQ020", ""),
    ("blood pressure, hypertension", "BPQ", "BPQ030", ""),
    ("blood pressure, hypertension", "BPQ", "BPD035", ""),
    ("blood pressure, hypertension", "BPQ", "BPQ040A",
     "antihypertensive (not diabetes) medication; some (ACE inhibitors/ARBs) are also given for "
     "diabetic kidney protection"),
    ("blood pressure, hypertension", "BPQ", "BPQ050A", "as BPQ040A"),
    ("blood pressure, hypertension", "BPX", "BPXPLS", ""),
    ("blood pressure, hypertension", "BPQ", "BPQ060", "healthcare contact (cholesterol test)"),
    ("blood pressure, hypertension", "BPQ", "BPQ080",
     "diagnosis that rests on a lab test (cholesterol); not glucose, but lab-derived"),
    ("blood pressure, hypertension", "BPQ", "BPQ090D", "lipid medication: lab-derived diagnosis"),
    # reproductive history
    *[("reproductive history", "RHQ", v, "") for v in [
        "RHQ010", "RHQ031", "RHQ060", "RHQ131", "RHQ160", "RHQ171", "RHD180", "RHQ172",
        "RHD280", "RHQ540"]],
    ("reproductive history", "RHQ", "RHQ162",
     "gestational diabetes is an ADA risk item, but it is a past glucose-based diabetes "
     "diagnosis; women who later developed diabetes may recall it more (borderline)"),
    ("reproductive history", "DEMO", "RIDEXPRG",
     "pregnancy at exam: HbA1c is lower in pregnancy; candidate EXCLUSION, not a feature"),
    # anything else: body measures, fitness, general health, mental health, comorbidities
    *[("other: body measures and fitness", "BMX", v, "") for v in [
        "BMXWT", "BMXARMC", "BMXARML", "BMXLEG", "BMDAVSAD"]],
    ("other: body measures and fitness", "MGX", "MGDCGSZ", ""),
    ("other: general and mental health", "HUQ", "HUQ010", ""),
    *[("other: general and mental health", "DPQ", v, "") for v in [
        "DPQ010", "DPQ020", "DPQ030", "DPQ040", "DPQ050", "DPQ060", "DPQ070", "DPQ080", "DPQ090"]],
    *[("other: general and mental health", "CDQ", v, "") for v in ["CDQ001", "CDQ010"]],
    ("other: healthcare access", "HUQ", "HUQ030",
     "healthcare access: drives being diagnosed vs undiagnosed (selection into the label)"),
    ("other: healthcare access", "HUQ", "HUQ051", "as HUQ030"),
    ("other: healthcare access", "HUQ", "HUQ071", "as HUQ030"),
    ("other: comorbidities and advice", "MCQ", "MCQ080",
     "doctor told overweight: follows a healthcare visit; mostly restates BMI"),
    ("other: comorbidities and advice", "MCQ", "MCQ300A", ""),
    *[("other: comorbidities and advice", "MCQ", v,
       "cardiovascular disease is a diabetes complication (reverse causation) and implies "
       "healthcare contact") for v in ["MCQ160B", "MCQ160C", "MCQ160E", "MCQ160F"]],
    ("other: comorbidities and advice", "MCQ", "MCQ160L", "liver disease (NAFLD link); diagnosis"),
    ("other: comorbidities and advice", "MCQ", "MCQ220", ""),
    *[("other: comorbidities and advice", "MCQ", v,
       "doctor advice (lose weight / exercise / salt / fat) can follow abnormal glucose or HbA1c "
       "results: possible label leakage") for v in ["MCQ365A", "MCQ365B", "MCQ365C", "MCQ365D"]],
    *[("other: comorbidities and advice", "MCQ", v,
       "behaviour change after the advice above: same leakage risk") for v in [
        "MCQ370A", "MCQ370B", "MCQ370C", "MCQ370D"]],
    ("other: comorbidities and advice", "KIQ_U", "KIQ022",
     "kidney disease is a diabetes complication (reverse causation)"),
    ("other: comorbidities and advice", "KIQ_U", "KIQ480",
     "nocturia: a symptom of hyperglycaemia (legitimate screening symptom, but close to the "
     "outcome)"),
    ("other: diabetes-adjacent questions", "DIQ", "DIQ160",
     "PREDIABETES DIAGNOSIS: based on a glucose/HbA1c test: near-direct proxy for the label"),
    ("other: diabetes-adjacent questions", "DIQ", "DIQ170",
     "told at risk for diabetes: follows testing or a clinical judgement: label-adjacent"),
    ("other: diabetes-adjacent questions", "DIQ", "DIQ172",
     "feels at risk: may reflect known test results: label-adjacent"),
    ("other: diabetes-adjacent questions", "DIQ", "DIQ180",
     "had a blood test for diabetes in past 3 years: healthcare contact, and a negative test "
     "makes undiagnosed diabetes less likely"),
]

# Gate for "asked of a subset" (pandas query on the merged frame). Verified, not assumed.
GATES = {
    "PAQ610": "PAQ605 == 1", "PAD615": "PAQ605 == 1", "PAQ625": "PAQ620 == 1",
    "PAD630": "PAQ620 == 1", "PAQ640": "PAQ635 == 1", "PAD645": "PAQ635 == 1",
    "PAQ655": "PAQ650 == 1", "PAD660": "PAQ650 == 1", "PAQ670": "PAQ665 == 1",
    "PAD675": "PAQ665 == 1",
    "SMQ040": "SMQ020 == 1", "SMD030": "SMQ020 == 1", "SMD641": "SMQ040 in [1, 2]",
    "SMD650": "SMQ040 in [1, 2]", "SMQ050Q": "SMQ040 == 3", "SMQ050U": "SMQ040 == 3",
    "ALQ110": "ALQ101 == 2", "ALQ120Q": "ALQ101 == 1 or ALQ110 == 1",
    "ALQ130": "ALQ120Q > 0 and ALQ120Q < 777", "ALQ141Q": "ALQ120Q > 0 and ALQ120Q < 777",
    "RHQ131": "RIAGENDR == 2", "RHQ160": "RHQ131 == 1", "RHQ162": "RHQ131 == 1",
    "RHQ171": "RHQ131 == 1", "RHD180": "RHQ131 == 1", "RHQ172": "RHQ131 == 1",
    "RHQ010": "RIAGENDR == 2", "RHQ031": "RIAGENDR == 2", "RHD280": "RIAGENDR == 2",
    "RHQ540": "RIAGENDR == 2", "RHQ060": "RHQ031 == 2",
    "BPQ030": "BPQ020 == 1", "BPD035": "BPQ020 == 1", "BPQ040A": "BPQ020 == 1",
    "BPQ050A": "BPQ040A == 1",
    "WHQ150": "WHD140 > 0 and WHD140 < 7777",
    "OCQ180": "OCD150 in [1, 2]", "DMDYRSUS": "DMDBORN4 == 2",
}

# Variables of the CURRENT pipeline (and the label) whose names matter for a 2005-2016 pool.
CORE = {"DEMO": ["RIAGENDR", "RIDAGEYR", "RIDRETH1", "RIDRETH3", "WTMEC2YR", "DMDBORN",
                 "DMDBORN2", "DMDBORN4", "INDHHINC", "INDHHIN2", "INDFMPIR", "DMDEDUC2"],
        "BMX": ["BMXBMI", "BMXWAIST", "BMXHT", "BMXWT", "BMDAVSAD"],
        "BPX": ["BPXSY1", "BPXDI1", "BPXPLS"],
        "DIQ": ["DIQ010", "DIQ050", "DIQ070", "DIQ160", "DIQ180"],
        "MCQ": ["MCQ300C", "MCQ365A"], "SMQ": ["SMQ020", "SMQ040"],
        "GHB": ["LBXGH"], "GLU": ["LBXGLU", "WTSAF2YR"],
        "SLQ": ["SLD010H", "SLD012"], "ALQ": ["ALQ101", "ALQ140Q", "ALQ141Q"],
        "PAQ": ["PAD020", "PAQ100", "PAD200", "PAD320", "PAQ605", "PAQ650", "PAD680"],
        "RHQ": ["RHQ131", "RHQ162"], "FSQ": ["FSDHH", "FSDAD"],
        "HUQ": ["HUQ050", "HUQ051"]}


def _get(stem: str, letter: str, ext: str) -> bytes | None:
    start = next(c for c, v in ALL.items() if v == letter).split("-")[0]
    r = requests.get(URL.format(start=start, stem=stem, letter=letter, ext=ext), timeout=180)
    return r.content if r.status_code == 200 else None


def fetch(stem: str, letter: str) -> Path | None:
    """XPT file, cached; None if CDC has no such file for that cycle (HTML or 404)."""
    path = RAW / f"{stem}_{letter}.XPT"
    if path.exists():
        with open(path, "rb") as f:
            if f.read(len(XPT_MAGIC)) == XPT_MAGIC:
                return path
    content = _get(stem, letter, "xpt")
    if not content or not content.startswith(XPT_MAGIC):
        return None
    path.write_bytes(content)
    print("downloaded", path.name, f"{len(content) / 1e6:.1f} MB", flush=True)
    return path


def codebook(stem: str, letter: str) -> dict[str, dict]:
    """{var: {label, target, dk_codes}} parsed from the CDC codebook page (cached)."""
    DOCS.mkdir(parents=True, exist_ok=True)
    path = DOCS / f"{stem}_{letter}.htm"
    if not path.exists():
        content = _get(stem, letter, "htm")
        if not content:
            return {}
        path.write_bytes(content)
    text = path.read_text(encoding="utf-8", errors="replace")
    out = {}
    for m in re.finditer(r'<h3 class="vartitle" id="([A-Za-z0-9_]+)">(.*?)(?=<h3 class="vartitle"|\Z)',
                         text, re.S):
        var, body = m.group(1).upper(), m.group(2)   # MCQ pages use e.g. MCQ160b
        lab = re.search(r"SAS Label: </dt>\s*<dd>(.*?)</dd>", body, re.S)
        targets = [" ".join(html.unescape(t).split())
                   for t in re.findall(r"Target: </dt>\s*<dd>(.*?)</dd>", body, re.S)]
        rows = re.findall(r'<td scope="row" class="values">(.*?)</td>\s*<td class="values">(.*?)</td>',
                          body, re.S)
        dk = [c.strip() for c, d in rows if d.strip().lower() in ("refused", "don't know")]
        out[var] = {"label": html.unescape(lab.group(1).strip()) if lab else "",
                    "target": "; ".join(targets), "dk_codes": dk}
    return out


def analysis_seqn(letter: str) -> pd.DataFrame:
    """SEQN + age/sex of the development analysis sample for one cycle (same rules as the label)."""
    df = pd.read_sas(fetch("DEMO", letter), format="xport")[["SEQN", "RIAGENDR", "RIDAGEYR"]]
    for stem, cols in [("DIQ", ["DIQ010", "DIQ050", "DIQ070"]), ("GHB", ["LBXGH"]),
                       ("GLU", ["LBXGLU", "WTSAF2YR"])]:
        df = df.merge(pd.read_sas(fetch(stem, letter), format="xport")[["SEQN", *cols]],
                      on="SEQN", how="left")
    return label_frame(df)[["SEQN", "RIAGENDR", "RIDAGEYR"]]


def _codes(dk: list[str]) -> list[float]:
    vals = []
    for c in dk:
        try:
            vals.append(float(c))
        except ValueError:
            pass
    return vals


def audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stems = sorted({s for _, s, _, _ in SPEC} | set(CORE))
    # ---- 1. development cycles: merged frame restricted to the analysis sample
    frames = []
    for cycle, L in DEV.items():
        base = analysis_seqn(L).assign(cycle=cycle)
        for stem in stems:
            if stem in ("GHB", "GLU"):
                continue                        # lab files: not candidate features
            p = fetch(stem, L)
            if p is None:
                continue
            d = pd.read_sas(p, format="xport")
            keep = [v for _, s, v, _ in SPEC if s == stem and v in d] + \
                   [v for v in GATE_VARS if v in d and v not in base]
            keep = list(dict.fromkeys(keep))
            if stem == "DR1TOT":                    # one row per person
                d = d.drop_duplicates("SEQN")
            base = base.merge(d[["SEQN", *[k for k in keep if k not in base]]], on="SEQN",
                              how="left")
        frames.append(base)
    dev = pd.concat(frames, ignore_index=True)
    print("analysis sample n =", len(dev), dict(dev.cycle.value_counts().sort_index()), flush=True)

    books = {(s, L): codebook(s, L) for s in stems for L in DEV.values()}
    rows = []
    for group, stem, var, flag in SPEC:
        cycles = [c for c, L in DEV.items() if var in dev and dev.loc[dev.cycle == c, var].notna().any()]
        info = next((books[(stem, L)][var] for L in ("I", "H", "G") if var in books[(stem, L)]), {})
        sub = dev[dev.cycle.isin(cycles)]
        if not cycles:
            rows.append({"group": group, "file": stem, "variable": var,
                         "meaning": info.get("label", ""), "cycles": "none in 2011-2016",
                         "missing_pct": np.nan, "refused_dk_pct": np.nan,
                         "target": info.get("target", ""), "universe": "", "flag": flag})
            continue
        x = sub[var]
        dk = x.isin(_codes(info.get("dk_codes", []))) if info.get("dk_codes") else x.isna() & False
        target = info.get("target", "")
        sex_only = "Females only" in target or "Males only" in target
        eligible = sub.RIAGENDR == 2 if "Females only" in target else (
            sub.RIAGENDR == 1 if "Males only" in target else pd.Series(True, index=sub.index))
        miss_elig = x[eligible].isna().mean()
        if var in GATES:
            try:
                g = sub.eval(GATES[var]).fillna(False).astype(bool)
                a_in, a_out = x[g].notna().mean(), x[~g].notna().mean()
                universe = (f"subset: asked if {GATES[var]} (answered {a_in:.0%} when true, "
                            f"{a_out:.0%} when false)")
            except Exception as e:                # gate variable absent in some cycle
                universe = f"subset (gate {GATES[var]} not checkable: {e.__class__.__name__})"
        elif miss_elig < 0.05:
            universe = "everyone in target" + (" (one sex)" if sex_only else "")
        else:
            universe = f"subset or non-response: {miss_elig:.0%} missing among eligible; gate not checked"
        rows.append({"group": group, "file": stem, "variable": var, "meaning": info.get("label", ""),
                     "cycles": ", ".join(c[2:4] + "-" + c[7:] for c in cycles) if len(cycles) < 3
                     else "all 3",
                     "missing_pct": 100 * x.isna().mean(), "refused_dk_pct": 100 * dk.mean(),
                     "target": target, "universe": universe, "flag": flag})
    features = pd.DataFrame(rows)

    # ---- 2. HbA1c in 2005-2010 vs 2011-2016, with the main label's sample rules applied
    hba = []
    for cycle, L in ALL.items():
        p = fetch("GHB", L)
        cb = codebook("GHB", L).get("LBXGH", {})
        d = pd.read_sas(fetch("DEMO", L), format="xport")[["SEQN", "RIDAGEYR", "WTMEC2YR"]]
        diq = pd.read_sas(fetch("DIQ", L), format="xport")
        pills = "DID070" if "DID070" in diq else "DIQ070"   # pills: DID070 in 2005-2008
        diq = diq.rename(columns={"DID070": "DIQ070"})
        d = d.merge(diq[["SEQN", "DIQ010", "DIQ050", "DIQ070"]], on="SEQN", how="left")
        d = d.merge(pd.read_sas(p, format="xport")[["SEQN", "LBXGH"]], on="SEQN", how="left")
        d = d.merge(pd.read_sas(fetch("GLU", L), format="xport")[["SEQN", "LBXGLU", "WTSAF2YR"]],
                    on="SEQN", how="left")
        s = label_frame(d)
        w = s.WTMEC2YR
        hba.append({"cycle": cycle, "file": f"GHB_{L}", "variable": "LBXGH",
                    "label": cb.get("label", ""), "target": cb.get("target", ""),
                    "pills_variable": pills,
                    "analysis_n": len(s), "positives": int(s.undiagnosed_diabetes.sum()),
                    "prevalence_unweighted_pct": 100 * s.undiagnosed_diabetes.mean(),
                    "prevalence_weighted_pct": 100 * (w * s.undiagnosed_diabetes).sum() / w.sum()})
    hba = pd.DataFrame(hba)

    # ---- 3. name availability 2005-2016 for core + candidate variables
    names = []
    by_stem: dict[str, list[str]] = {}
    for _, s, v, _ in SPEC:
        by_stem.setdefault(s, []).append(v)
    for s, vs in CORE.items():
        by_stem.setdefault(s, []).extend(vs)
    for stem, vs in by_stem.items():
        cols = {}
        for cycle, L in ALL.items():
            p = fetch(stem, L)
            cols[cycle] = set(pd.read_sas(p, format="xport", chunksize=1).read(1).columns) if p else None
        for v in dict.fromkeys(vs):
            names.append({"file": stem, "variable": v,
                          **{c: ("no file" if cols[c] is None else ("yes" if v in cols[c] else "-"))
                             for c in ALL}})
    names = pd.DataFrame(names)
    return features, hba, names


GATE_VARS = sorted({"RIAGENDR", "RIDAGEYR", "BPQ020", "BPQ040A", "BPQ060", "BPQ080", "SMQ020",
                    "SMQ040", "ALQ101", "ALQ110", "ALQ120Q", "RHQ131", "RHQ031", "PAQ605",
                    "PAQ620", "PAQ635", "PAQ650", "PAQ665", "WHD140", "MCQ365A", "OCD150",
                    "DMDBORN4"})


def md(df: pd.DataFrame) -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v).replace("|", "/") for v in r) + " |"
                            for r in df.itertuples(index=False))


START, END = "<!-- GENERATED TABLES START -->", "<!-- GENERATED TABLES END -->"


def tables_markdown(features: pd.DataFrame, hba: pd.DataFrame, names: pd.DataFrame) -> str:
    parts = ["## A. Candidate features by group (2011-2016 development sample, n = 13,355)\n",
             "Missing % = no value in the cycles where the variable exists (structural skips "
             "included). Refused/DK % = coded refused or don't know (codes read from the "
             "codebook). ⚠ = borderline, for you to decide.\n"]
    for group, g in features.groupby("group", sort=False):
        t = g.assign(flag=g.flag.map(lambda f: f"⚠ {f}" if f else ""))
        parts += [f"\n### {group}\n", md(t[["file", "variable", "meaning", "cycles", "missing_pct",
                                            "refused_dk_pct", "target", "universe", "flag"]]), ""]
    parts += ["\n## B. HbA1c by cycle, main-label sample rules applied (unweighted and "
              "WTMEC2YR-weighted)\n", md(hba), ""]
    parts += ["\n## C. Variable availability 2005-2016 (yes = present, - = absent, "
              "no file = component not fielded)\n", md(names), ""]
    return "\n".join(parts)


if __name__ == "__main__":
    features, hba, names = audit()
    Path("docs").mkdir(exist_ok=True)
    features.to_csv("docs/feature_audit_features.csv", index=False, float_format="%.1f")
    names.to_csv("docs/feature_audit_names.csv", index=False)
    hba.to_csv("docs/feature_audit_hba1c.csv", index=False, float_format="%.2f")
    doc = Path("docs/feature_audit.md")
    if doc.exists() and START in doc.read_text(encoding="utf-8"):
        text = doc.read_text(encoding="utf-8")
        head, rest = text.split(START, 1)
        tail = rest.split(END, 1)[1]
        doc.write_text(f"{head}{START}\n{tables_markdown(features, hba, names)}\n{END}{tail}",
                       encoding="utf-8")
        print("tables written into", doc)
    print(md(hba))
