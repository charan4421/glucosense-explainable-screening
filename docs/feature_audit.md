# Feature audit: non-lab screening candidates, NHANES 2011-2016

Audit only: no models were run and no 2017-2018 file was opened. Regenerate the tables with
`python scripts/audit_features.py` (it downloads the public XPT files and CDC codebook pages into
the gitignored `data/raw/`). The tables are also in CSV form next to this file
(`feature_audit_features.csv`, `feature_audit_hba1c.csv`, `feature_audit_names.csv`).

**How the columns were filled**

- **Meaning** and **target** (age/sex universe) come from the CDC codebook page of each file
  (SAS label and "Target").
- **Missing %** is measured on our development analysis sample: adults ≥ 20 in 2011-2016 after
  the label exclusions, n = 13,355 (4,338 / 4,666 / 4,351). It is computed only in the cycles
  where the variable exists, and it includes structural skips.
- **Refused/DK %** uses the "Refused" and "Don't know" codes listed in each variable's codebook
  table.
- **Universe**:
  - "Everyone in target" means under 5% missing among eligible people.
  - "Subset: asked if …" names a skip-pattern gate that was **checked on the data**. The two
    percentages are the share answering when the gate is true vs false.
  - "Subset or non-response" means the gate was not checked.
  - Three gates I first assumed (BPQ080, BPQ090D, MCQ370A) were wrong on the data and were
    removed.

**Excluded by rule, not listed:** every laboratory result (GHB, GLU, OGTT, insulin, lipids, urine,
and all `LB*` variables); diabetes diagnosis, insulin and pill items (DIQ010, DID040, DIQ050,
DIQ070/DID070, DIQ060, and the DIQ care items); and the prescription file RXQ_RX. RXQ_RX contains
diabetes drugs; non-diabetes drug classes could be derived from it, but that needs a separate
review.

## Borderline items, for you to decide

These are listed in the tables with ⚠. I have not decided any of them.

| Variable(s) | Why it is borderline |
|---|---|
| **DIQ160** ever told prediabetes | Rests on a glucose or HbA1c test: a near-direct proxy for the label |
| **DIQ170** told at risk, **DIQ172** feels at risk | Follow testing or a clinical judgement: label-adjacent |
| **DIQ180** blood test for diabetes in past 3 years | Healthcare contact; a recent negative test makes undiagnosed diabetes less likely |
| **MCQ365A-D** doctor advised to lose weight, exercise, or cut salt/fat; **MCQ370A-D** now doing it | Advice can follow abnormal glucose or HbA1c results: possible label leakage |
| **HIQ011** insurance, **HUQ030** routine place of care, **HUQ051** healthcare visits, **HUQ071** hospital stay | Healthcare access decides who is diagnosed vs undiagnosed, so it shapes the label itself |
| **BPQ080** told high cholesterol, **BPQ090D** lipid medication, **BPQ060** cholesterol ever checked | Diagnosis rests on a lab test (not glucose), and implies healthcare contact |
| **BPQ040A / BPQ050A** antihypertensive medication | Not a diabetes drug, but ACE inhibitors/ARBs are also given for diabetic kidney protection |
| **MCQ160B/C/E/F** heart failure, CHD, heart attack, stroke; **KIQ022** weak/failing kidneys; **MCQ160L** liver condition | Complications of diabetes (reverse causation), and imply healthcare contact |
| **KIQ480** night-time urination | A symptom of hyperglycaemia: a legitimate screening symptom, but close to the outcome |
| **WHQ060** weight change intentional | Unintentional weight loss is a symptom of hyperglycaemia |
| **RHQ162** gestational diabetes | An official ADA item, but it is a past glucose-based diagnosis, and women who later developed diabetes may recall it more |
| **MCQ080** doctor said overweight | Follows a healthcare visit and mostly restates BMI |
| **RIDEXPRG** pregnant at exam | HbA1c is lower in pregnancy: a candidate **exclusion**, not a feature (pregnant women are currently kept) |
| **DR1TOT** 24-h recall totals | Not label-related, but a single day is noisy for one person, and the file has its own dietary weight (WTDRD1) |

## Data issues inside 2011-2016 (relevant even without pooling)

1. **Fasting glucose instrument change in 2015-2016 (affects our sensitivity label).**
   - CDC (GLU_I note): the new Roche Cobas C311 reads about 2% higher than the C501 used before.
   - Among adults without diagnosed diabetes, FPG ≥ 126 was 5.5% on the new instrument vs 3.4%
     back-adjusted.
   - CDC gives a backward equation: C501 = 0.9776 × C311 + 0.4994.
   - In our 2015-2016 fasting sample, **29 of 147 sensitivity-label positives (20%) disappear**
     after back-adjustment (7.21% → 5.78%). 2011-2014 are unchanged.
   - The main HbA1c label is not affected.
2. **The pill question reached more people in 2015-2016.** Among adults answering "no"
   diagnosis, DIQ070 was asked of about 5% in 2011-2014 and 10% in 2015-2016 (yes: 16 / 19 / 31).
   Our exclusion of "pills without a diagnosis" therefore removes more people in 2015-2016.
3. **Sleep was redesigned in 2015-2016.**
   - SLD010H (2011-2014) became SLD012 (2015-2016), with different wording.
   - SLQ030/SLQ040/SLQ120 exist only in 2015-2016.
   - SLQ060 exists only in 2011-2014.
4. **HUQ050 (2011-2012) became HUQ051 (2013+).** I did not check whether the categories match.
5. **Grip strength (MGX) exists only in 2011-2014.**
6. **Skip patterns are not "missing".** Physical activity, smoking, alcohol, blood pressure and
   reproductive follow-ups are 57-82% "missing" because people answered "no" to the gate
   question. They must be derived (e.g. MET-minutes = 0, pack-years = 0), not KNN-imputed.
7. **The PHQ-9 (DPQ, about 10% missing) and the dietary recall (about 8% missing)** have real
   non-response on top of the skip patterns.

## HbA1c in 2005-2006, 2007-2008 and 2009-2010

**It is the same variable with the same definition, but not the same level.**

- **Same variable:** LBXGH, "Glycohemoglobin (%)", target 12+ years, in all six cycles. Our label
  rules run unchanged, except that the pill item is named DID070 in 2005-2008.
- **Lab and instrument history**, from the CDC GHB documentation:
  - 2005-2006: Tosoh A1c 2.2 Plus.
  - 2007-2010: Tosoh A1c G7. Both at the University of Minnesota (Fairview) laboratory.
  - 2013-2014: testing moved to the University of Missouri, with "no changes to the lab method,
    or lab equipment".
  - 2015-2016: "no changes to the lab method, lab equipment, or lab site". The method file is
    titled "Tosoh G8".
  - All laboratories met NGSP bias and precision criteria.
- **CDC analytic note:**
  - An **unexplained upward shift** in HbA1c from 2005-2006 to 2007-2010. HbA1c ≥ 6.5% rose
    from 5.8% to 7.3% (adults ≥ 18, weighted, age-adjusted, including diagnosed diabetes).
  - The cause (laboratory, survey design or population) could not be determined.
  - The data were re-released unchanged, the earlier cross-over regression was withdrawn, and
    users are told to "carefully consider" this.
- **With our label rules applied** (table B): undiagnosed prevalence is
  - 2.4% (2005-2006)
  - 3.7% (2007-2008)
  - 3.4% (2009-2010)
  - 3.5% / 2.8% / 2.9% (2011-2016)

  The 2005-2006 → 2007-2008 jump is the documented shift.
- **Implication:** 2005-2006 is not level-comparable. 2007-2010 is closer, but it sits inside the
  period CDC flags.
- **Decision for you:**
  - pool 2007-2016 and treat cycle as a sensitivity check, or
  - stay with 2011-2016.

## Variable-name and definition changes that make a pooled 2005-2016 design hard

Each item is checked against the file columns (table C). Where categories may differ, I note that
I did not verify them.

1. **Race/ethnicity: RIDRETH3 (with a Non-Hispanic Asian category) exists only from 2011.**
   - 2005-2010 has only RIDRETH1, which puts Asian adults in "Other".
   - Pooling would lose the group with the largest calibration error in our subgroup analysis
     (O/E 1.31).
2. **Diabetes pills (label exclusion): DID070 in 2005-2008, DIQ070 from 2009.** The meaning is
   the same ("Take diabetic pills to lower blood sugar"), and the skip pattern is similar.
3. **Physical activity was a different instrument in 2005-2006** (PAD020, PAQ100, PAD200,
   PAD320). The GPAQ-style items (PAQ605-PAD680) start in 2007. It is not harmonisable without a
   mapping.
4. **Sleep: SLD010H in 2005-2014, then SLD012 in 2015-2016.** SLQ030/040/120 exist only in
   2005-2008 and 2015-2016.
5. **Binge drinking: ALQ140Q (2005-2010, "5 or more drinks") became ALQ141Q (2011+, "4/5
   drinks").** The threshold changed. ALQ151 exists only from 2011.
6. **Country of birth: DMDBORN (2005-2006) → DMDBORN2 (2007-2010) → DMDBORN4 (2011+).** I did
   not verify whether the categories match.
7. **Household income: INDHHINC (2005-2006) → INDHHIN2 (2007+).** The bands differ. INDFMPIR is
   stable and is the safer choice.
8. **Healthcare visits: HUQ050 (2005-2012) → HUQ051 (2013+).**
9. **Absent before 2011:** sagittal abdominal diameter (BMDAVSAD), doctor advice (MCQ365/370),
   DIQ172, TV/computer hours (PAQ710/715).
   - Grip strength exists only in 2011-2014.
   - **Absent in 2005-2006:** gestational diabetes (RHQ162, so the ADA-style GDM item is
     impossible there), RHQ172, RHD180, BPD035, DBD895-910, FSQ165.
10. **Lab changes that move the labels:**
    - HbA1c: the unexplained 2007 shift (above).
    - Fasting glucose:
      - Roche/Hitachi 911 → Modular P in 2007 (CDC: ModP = 911 + 1.148 mg/dL).
      - Then C501 → C311 in 2015-2016 (about +2%).

<!-- GENERATED TABLES START -->
## A. Candidate features by group (2011-2016 development sample, n = 13,355)

Missing % = no value in the cycles where the variable exists (structural skips included). Refused/DK % = coded refused or don't know (codes read from the codebook). ⚠ = borderline, for you to decide.


### physical activity

| file | variable | meaning | cycles | missing_pct | refused_dk_pct | target | universe | flag |
|---|---|---|---|---|---|---|---|---|
| PAQ | PAQ605 | Vigorous work activity | all 3 | 0.0 | 0.0 | Both males and females 12 YEARS - 150 YEARS | everyone in target |  |
| PAQ | PAQ610 | Number of days vigorous work | all 3 | 80.7 | 0.0 | Both males and females 12 YEARS - 150 YEARS | subset: asked if PAQ605 == 1 (answered 100% when true, 0% when false) |  |
| PAQ | PAD615 | Minutes vigorous-intensity work | all 3 | 80.8 | 0.0 | Both males and females 12 YEARS - 150 YEARS | subset: asked if PAQ605 == 1 (answered 99% when true, 0% when false) |  |
| PAQ | PAQ620 | Moderate work activity | all 3 | 0.0 | 0.0 | Both males and females 12 YEARS - 150 YEARS | everyone in target |  |
| PAQ | PAQ625 | Number of days moderate work | all 3 | 64.7 | 0.1 | Both males and females 12 YEARS - 150 YEARS | subset: asked if PAQ620 == 1 (answered 100% when true, 0% when false) |  |
| PAQ | PAD630 | Minutes moderate-intensity work | all 3 | 64.9 | 0.1 | Both males and females 12 YEARS - 150 YEARS | subset: asked if PAQ620 == 1 (answered 100% when true, 0% when false) |  |
| PAQ | PAQ635 | Walk or bicycle | all 3 | 0.0 | 0.0 | Both males and females 12 YEARS - 150 YEARS | everyone in target |  |
| PAQ | PAQ640 | Number of days walk or bicycle | all 3 | 72.9 | 0.0 | Both males and females 12 YEARS - 150 YEARS | subset: asked if PAQ635 == 1 (answered 100% when true, 0% when false) |  |
| PAQ | PAD645 | Minutes walk/bicycle for transportation | all 3 | 73.0 | 0.1 | Both males and females 12 YEARS - 150 YEARS | subset: asked if PAQ635 == 1 (answered 100% when true, 0% when false) |  |
| PAQ | PAQ650 | Vigorous recreational activities | all 3 | 0.0 | 0.0 | Both males and females 12 YEARS - 150 YEARS | everyone in target |  |
| PAQ | PAQ655 | Days vigorous recreational activities | all 3 | 74.7 | 0.0 | Both males and females 12 YEARS - 150 YEARS | subset: asked if PAQ650 == 1 (answered 100% when true, 0% when false) |  |
| PAQ | PAD660 | Minutes vigorous recreational activities | all 3 | 74.7 | 0.0 | Both males and females 12 YEARS - 150 YEARS | subset: asked if PAQ650 == 1 (answered 100% when true, 0% when false) |  |
| PAQ | PAQ665 | Moderate recreational activities | all 3 | 0.0 | 0.0 | Both males and females 12 YEARS - 150 YEARS | everyone in target |  |
| PAQ | PAQ670 | Days moderate recreational activities | all 3 | 57.3 | 0.0 | Both males and females 12 YEARS - 150 YEARS | subset: asked if PAQ665 == 1 (answered 100% when true, 0% when false) |  |
| PAQ | PAD675 | Minutes moderate recreational activities | all 3 | 57.4 | 0.0 | Both males and females 12 YEARS - 150 YEARS | subset: asked if PAQ665 == 1 (answered 100% when true, 0% when false) |  |
| PAQ | PAD680 | Minutes sedentary activity | all 3 | 0.1 | 0.5 | Both males and females 12 YEARS - 150 YEARS | everyone in target |  |
| PAQ | PAQ710 | Hours watch TV or videos past 30 days | all 3 | 0.0 | 0.1 | Both males and females 2 YEARS - 150 YEARS | everyone in target |  |
| PAQ | PAQ715 | Hours use computer past 30 days | all 3 | 0.0 | 0.0 | Both males and females 2 YEARS - 150 YEARS | everyone in target |  |


### sleep

| file | variable | meaning | cycles | missing_pct | refused_dk_pct | target | universe | flag |
|---|---|---|---|---|---|---|---|---|
| SLQ | SLD010H | How much sleep do you get (hours)? | 11-12, 13-14 | 0.1 | 0.0 | Both males and females 16 YEARS - 150 YEARS | everyone in target |  |
| SLQ | SLD012 | Sleep hours | 15-16 | 0.5 | 0.0 | Both males and females 16 YEARS - 150 YEARS | everyone in target |  |
| SLQ | SLQ030 | How often do you snore? | 15-16 | 0.0 | 7.3 | Both males and females 16 YEARS - 150 YEARS | everyone in target |  |
| SLQ | SLQ040 | How often do you snort or stop breathing | 15-16 | 0.0 | 5.8 | Both males and females 16 YEARS - 150 YEARS | everyone in target |  |
| SLQ | SLQ050 | Ever told doctor had trouble sleeping? | all 3 | 0.0 | 0.0 | Both males and females 16 YEARS - 150 YEARS | everyone in target |  |
| SLQ | SLQ060 | Ever told by doctor have sleep disorder? | 11-12, 13-14 | 0.0 | 0.1 | Both males and females 16 YEARS - 150 YEARS | everyone in target |  |
| SLQ | SLQ120 | How often feel overly sleepy during day? | 15-16 | 0.0 | 0.0 | Both males and females 16 YEARS - 150 YEARS | everyone in target |  |


### smoking

| file | variable | meaning | cycles | missing_pct | refused_dk_pct | target | universe | flag |
|---|---|---|---|---|---|---|---|---|
| SMQ | SMQ020 | Smoked at least 100 cigarettes in life | all 3 | 0.0 | 0.1 | Both males and females 18 YEARS - 150 YEARS | everyone in target |  |
| SMQ | SMQ040 | Do you now smoke cigarettes? | all 3 | 58.2 | 0.0 | Both males and females 18 YEARS - 150 YEARS | subset: asked if SMQ020 == 1 (answered 100% when true, 0% when false) |  |
| SMQ | SMD030 | Age started smoking cigarettes regularly | all 3 | 58.2 | 0.2 | Both males and females 18 YEARS - 150 YEARS | subset: asked if SMQ020 == 1 (answered 100% when true, 0% when false) |  |
| SMQ | SMD641 | # days smoked cigs during past 30 days | all 3 | 79.6 | 0.0 | Both males and females 12 YEARS - 150 YEARS | subset: asked if SMQ040 in [1, 2] (answered 100% when true, 0% when false) |  |
| SMQ | SMD650 | Avg # cigarettes/day during past 30 days | all 3 | 79.8 | 0.0 | Both males and females 12 YEARS - 150 YEARS | subset: asked if SMQ040 in [1, 2] (answered 99% when true, 0% when false) |  |
| SMQ | SMQ050Q | How long since quit smoking cigarettes | all 3 | 78.6 | 0.1 | Both males and females 18 YEARS - 150 YEARS | subset: asked if SMQ040 == 3 (answered 100% when true, 0% when false) |  |
| SMQ | SMQ050U | Unit of measure (day/week/month/year) | all 3 | 79.3 | 0.0 | Both males and females 18 YEARS - 150 YEARS | subset: asked if SMQ040 == 3 (answered 97% when true, 0% when false) |  |


### alcohol

| file | variable | meaning | cycles | missing_pct | refused_dk_pct | target | universe | flag |
|---|---|---|---|---|---|---|---|---|
| ALQ | ALQ101 | Had at least 12 alcohol drinks/1 yr? | all 3 | 9.5 | 0.1 | Both males and females 18 YEARS - 150 YEARS | subset or non-response: 9% missing among eligible; gate not checked |  |
| ALQ | ALQ110 | Had at least 12 alcohol drinks/lifetime? | all 3 | 75.0 | 0.1 | Both males and females 18 YEARS - 150 YEARS | subset: asked if ALQ101 == 2 (answered 100% when true, 0% when false) |  |
| ALQ | ALQ120Q | How often drink alcohol over past 12 mos | all 3 | 23.1 | 0.1 | Both males and females 18 YEARS - 150 YEARS | subset: asked if ALQ101 == 1 or ALQ110 == 1 (answered 100% when true, 0% when false) |  |
| ALQ | ALQ120U | # days drink alcohol per wk, mo, yr | all 3 | 36.9 | 0.0 | Both males and females 18 YEARS - 150 YEARS | subset or non-response: 37% missing among eligible; gate not checked |  |
| ALQ | ALQ130 | Avg # alcoholic drinks/day - past 12 mos | all 3 | 36.8 | 0.1 | Both males and females 18 YEARS - 150 YEARS | subset: asked if ALQ120Q > 0 and ALQ120Q < 777 (answered 82% when true, 0% when false) |  |
| ALQ | ALQ141Q | # days have 4/5 drinks - past 12 mos | all 3 | 36.8 | 0.2 | Both males and females 18 YEARS - 150 YEARS | subset: asked if ALQ120Q > 0 and ALQ120Q < 777 (answered 82% when true, 0% when false) |  |
| ALQ | ALQ141U | # days per week, month, year? | all 3 | 73.5 | 0.0 | Both males and females 18 YEARS - 150 YEARS | subset or non-response: 73% missing among eligible; gate not checked |  |
| ALQ | ALQ151 | Ever have 4/5 or more drinks every day? | all 3 | 23.1 | 0.0 | Both males and females 18 YEARS - 150 YEARS | subset or non-response: 23% missing among eligible; gate not checked |  |


### diet quality

| file | variable | meaning | cycles | missing_pct | refused_dk_pct | target | universe | flag |
|---|---|---|---|---|---|---|---|---|
| DBQ | DBQ700 | How healthy is the diet | all 3 | 0.0 | 0.0 | Both males and females 16 YEARS - 150 YEARS | everyone in target |  |
| DBQ | DBD895 | # of meals not home prepared | all 3 | 0.0 | 0.0 | Both males and females 1 YEARS - 150 YEARS | everyone in target |  |
| DBQ | DBD900 | # of meals from fast food or pizza place | all 3 | 22.5 | 0.0 | Both males and females 1 YEARS - 150 YEARS | subset or non-response: 22% missing among eligible; gate not checked |  |
| DBQ | DBD905 | # of ready-to-eat foods in past 30 days | all 3 | 0.2 | 0.1 | Both males and females 1 YEARS - 150 YEARS | everyone in target |  |
| DBQ | DBD910 | # of frozen meals/pizza in past 30 days | all 3 | 0.2 | 0.0 | Both males and females 1 YEARS - 150 YEARS | everyone in target |  |
| DBQ | DBQ197 | Past 30 day milk product consumption | all 3 | 0.0 | 0.0 | Both males and females 1 YEARS - 150 YEARS | everyone in target |  |
| DR1TOT | DR1TKCAL | Energy (kcal) | all 3 | 8.2 | 0.0 | Both males and females 0 YEARS - 150 YEARS | subset or non-response: 8% missing among eligible; gate not checked | ⚠ single 24-h recall (day 1): noisy for one person; has its own dietary weight WTDRD1 |
| DR1TOT | DR1TCARB | Carbohydrate (gm) | all 3 | 8.2 | 0.0 | Both males and females 0 YEARS - 150 YEARS | subset or non-response: 8% missing among eligible; gate not checked | ⚠ single 24-h recall (day 1): noisy for one person; has its own dietary weight WTDRD1 |
| DR1TOT | DR1TSUGR | Total sugars (gm) | all 3 | 8.2 | 0.0 | Both males and females 0 YEARS - 150 YEARS | subset or non-response: 8% missing among eligible; gate not checked | ⚠ single 24-h recall (day 1): noisy for one person; has its own dietary weight WTDRD1 |
| DR1TOT | DR1TFIBE | Dietary fiber (gm) | all 3 | 8.2 | 0.0 | Both males and females 0 YEARS - 150 YEARS | subset or non-response: 8% missing among eligible; gate not checked | ⚠ single 24-h recall (day 1): noisy for one person; has its own dietary weight WTDRD1 |
| DR1TOT | DR1TTFAT | Total fat (gm) | all 3 | 8.2 | 0.0 | Both males and females 0 YEARS - 150 YEARS | subset or non-response: 8% missing among eligible; gate not checked | ⚠ single 24-h recall (day 1): noisy for one person; has its own dietary weight WTDRD1 |
| DR1TOT | DR1TSFAT | Total saturated fatty acids (gm) | all 3 | 8.2 | 0.0 | Both males and females 0 YEARS - 150 YEARS | subset or non-response: 8% missing among eligible; gate not checked | ⚠ single 24-h recall (day 1): noisy for one person; has its own dietary weight WTDRD1 |
| DR1TOT | DR1TSODI | Sodium (mg) | all 3 | 8.2 | 0.0 | Both males and females 0 YEARS - 150 YEARS | subset or non-response: 8% missing among eligible; gate not checked | ⚠ single 24-h recall (day 1): noisy for one person; has its own dietary weight WTDRD1 |
| DR1TOT | DR1TALCO | Alcohol (gm) | all 3 | 8.2 | 0.0 | Both males and females 0 YEARS - 150 YEARS | subset or non-response: 8% missing among eligible; gate not checked | ⚠ single 24-h recall (day 1): noisy for one person; has its own dietary weight WTDRD1 |
| DR1TOT | DR1DRSTZ | Dietary recall status | all 3 | 0.0 | 0.0 | Both males and females 0 YEARS - 150 YEARS | everyone in target | ⚠ single 24-h recall (day 1): noisy for one person; has its own dietary weight WTDRD1 |


### education, income, food security

| file | variable | meaning | cycles | missing_pct | refused_dk_pct | target | universe | flag |
|---|---|---|---|---|---|---|---|---|
| DEMO | DMDEDUC2 | Education level - Adults 20+ | all 3 | 0.0 | 0.0 | Both males and females 20 YEARS - 150 YEARS | everyone in target |  |
| DEMO | INDFMPIR | Ratio of family income to poverty | all 3 | 8.5 | 0.0 | Both males and females 0 YEARS - 150 YEARS | subset or non-response: 8% missing among eligible; gate not checked |  |
| DEMO | INDHHIN2 | Annual household income | all 3 | 1.6 | 3.5 | Both males and females 0 YEARS - 150 YEARS | everyone in target |  |
| DEMO | DMDMARTL | Marital status | all 3 | 0.0 | 0.0 | Both males and females 20 YEARS - 150 YEARS | everyone in target |  |
| DEMO | DMDHHSIZ | Total number of people in the Household | all 3 | 0.0 | 0.0 | Both males and females 0 YEARS - 150 YEARS | everyone in target |  |
| DEMO | DMDBORN4 | Country of birth | all 3 | 0.0 | 0.1 | Both males and females 0 YEARS - 150 YEARS | everyone in target |  |
| DEMO | DMDCITZN | Citizenship status | all 3 | 0.1 | 0.2 | Both males and females 0 YEARS - 150 YEARS | everyone in target |  |
| DEMO | DMDYRSUS | Length of time in US | all 3 | 68.9 | 1.2 | Both males and females 0 YEARS - 150 YEARS | subset: asked if DMDBORN4 == 2 (answered 100% when true, 0% when false) |  |
| DEMO | SIALANG | Language of SP Interview | all 3 | 0.0 | 0.0 | Both males and females 0 YEARS - 150 YEARS | everyone in target |  |
| FSQ | FSDHH | Household food security category | all 3 | 1.4 | 0.0 | Both males and females 0 YEARS - 150 YEARS | everyone in target |  |
| FSQ | FSDAD | Adult food security category | all 3 | 1.4 | 0.0 | Both males and females 0 YEARS - 150 YEARS | everyone in target |  |
| FSQ | FSQ165 | HH FS benefit: ever received | all 3 | 1.3 | 0.3 | Both males and females 0 YEARS - 150 YEARS | everyone in target |  |
| HIQ | HIQ011 | Covered by health insurance | all 3 | 0.0 | 0.1 | Both males and females 0 YEARS - 150 YEARS | everyone in target | ⚠ healthcare access: drives being diagnosed vs undiagnosed (selection into the label) |
| OCQ | OCD150 | Type of work done last week | all 3 | 0.0 | 0.0 | Both males and females 16 YEARS - 150 YEARS | everyone in target |  |
| OCQ | OCQ180 | Hours worked last week at all jobs | all 3 | 42.4 | 0.1 | Both males and females 16 YEARS - 150 YEARS | subset: asked if OCD150 in [1, 2] (answered 97% when true, 0% when false) |  |


### weight history

| file | variable | meaning | cycles | missing_pct | refused_dk_pct | target | universe | flag |
|---|---|---|---|---|---|---|---|---|
| WHQ | WHD010 | Current self-reported height (inches) | all 3 | 0.3 | 1.6 | Both males and females 16 YEARS - 150 YEARS | everyone in target |  |
| WHQ | WHD020 | Current self-reported weight (pounds) | all 3 | 0.3 | 1.1 | Both males and females 16 YEARS - 150 YEARS | everyone in target |  |
| WHQ | WHQ030 | How do you consider your weight | all 3 | 0.0 | 0.3 | Both males and females 16 YEARS - 150 YEARS | everyone in target |  |
| WHQ | WHQ040 | Like to weigh more, less or same | all 3 | 0.0 | 0.1 | Both males and females 16 YEARS - 150 YEARS | everyone in target |  |
| WHQ | WHD050 | Self-reported weight - 1 yr ago (pounds) | all 3 | 0.2 | 1.5 | Both males and females 16 YEARS - 150 YEARS | everyone in target |  |
| WHQ | WHD110 | Self-reported weight-10 yrs ago (pounds) | all 3 | 31.6 | 1.9 | Both males and females 36 YEARS - 150 YEARS | subset or non-response: 32% missing among eligible; gate not checked |  |
| WHQ | WHD120 | Self-reported weight-age 25 (pounds) | all 3 | 14.3 | 3.5 | Both males and females 27 YEARS - 150 YEARS | subset or non-response: 14% missing among eligible; gate not checked |  |
| WHQ | WHD130 | Self-reported height - age 25 (inches) | all 3 | 56.9 | 1.6 | Both males and females 50 YEARS - 150 YEARS | subset or non-response: 57% missing among eligible; gate not checked |  |
| WHQ | WHD140 | Self-reported greatest weight (pounds) | all 3 | 0.1 | 1.3 | Both males and females 18 YEARS - 150 YEARS | everyone in target |  |
| WHQ | WHQ150 | Age when heaviest weight | all 3 | 1.6 | 0.6 | Both males and females 18 YEARS - 150 YEARS | subset: asked if WHD140 > 0 and WHD140 < 7777 (answered 100% when true, 0% when false) |  |
| WHQ | WHQ070 | Tried to lose weight in past year | all 3 | 11.4 | 0.1 | Both males and females 16 YEARS - 150 YEARS | subset or non-response: 11% missing among eligible; gate not checked |  |
| WHQ | WHQ060 | Weight change intentional | all 3 | 82.2 | 0.0 | Both males and females 16 YEARS - 150 YEARS | subset or non-response: 82% missing among eligible; gate not checked | ⚠ unintentional weight loss is a symptom of hyperglycaemia (reverse causation) |


### blood pressure, hypertension

| file | variable | meaning | cycles | missing_pct | refused_dk_pct | target | universe | flag |
|---|---|---|---|---|---|---|---|---|
| BPQ | BPQ020 | Ever told you had high blood pressure | all 3 | 0.0 | 0.1 | Both males and females 16 YEARS - 150 YEARS | everyone in target |  |
| BPQ | BPQ030 | Told had high blood pressure - 2+ times | all 3 | 68.9 | 0.1 | Both males and females 16 YEARS - 150 YEARS | subset: asked if BPQ020 == 1 (answered 100% when true, 0% when false) |  |
| BPQ | BPD035 | Age told had hypertension | all 3 | 69.0 | 0.4 | Both males and females 16 YEARS - 150 YEARS | subset: asked if BPQ020 == 1 (answered 100% when true, 0% when false) |  |
| BPQ | BPQ040A | Taking prescription for hypertension | all 3 | 68.9 | 0.0 | Both males and females 16 YEARS - 150 YEARS | subset: asked if BPQ020 == 1 (answered 100% when true, 0% when false) | ⚠ antihypertensive (not diabetes) medication; some (ACE inhibitors/ARBs) are also given for diabetic kidney protection |
| BPQ | BPQ050A | Now taking prescribed medicine for HBP | all 3 | 74.2 | 0.0 | Both males and females 16 YEARS - 150 YEARS | subset: asked if BPQ040A == 1 (answered 100% when true, 0% when false) | ⚠ as BPQ040A |
| BPX | BPXPLS | 60 sec. pulse (30 sec. pulse * 2) | all 3 | 3.2 | 0.0 | Both males and females 8 YEARS - 150 YEARS | everyone in target |  |
| BPQ | BPQ060 | Ever had blood cholesterol checked | all 3 | 29.8 | 2.4 | Both males and females 16 YEARS - 150 YEARS | subset or non-response: 30% missing among eligible; gate not checked | ⚠ healthcare contact (cholesterol test) |
| BPQ | BPQ080 | Doctor told you - high cholesterol level | all 3 | 0.0 | 0.5 | Both males and females 16 YEARS - 150 YEARS | everyone in target | ⚠ diagnosis that rests on a lab test (cholesterol); not glucose, but lab-derived |
| BPQ | BPQ090D | Told to take prescriptn for cholesterol | all 3 | 26.8 | 0.1 | Both males and females 16 YEARS - 150 YEARS | subset or non-response: 27% missing among eligible; gate not checked | ⚠ lipid medication: lab-derived diagnosis |


### reproductive history

| file | variable | meaning | cycles | missing_pct | refused_dk_pct | target | universe | flag |
|---|---|---|---|---|---|---|---|---|
| RHQ | RHQ010 | Age when first menstrual period occurred | all 3 | 53.8 | 0.4 | Females only 12 YEARS - 150 YEARS | subset: asked if RIAGENDR == 2 (answered 89% when true, 0% when false) |  |
| RHQ | RHQ031 | Had regular periods in past 12 months | all 3 | 53.8 | 0.0 | Females only 12 YEARS - 150 YEARS | subset: asked if RIAGENDR == 2 (answered 89% when true, 0% when false) |  |
| RHQ | RHQ060 | Age at last menstrual period | all 3 | 77.3 | 0.8 | Females only 12 YEARS - 150 YEARS | subset: asked if RHQ031 == 2 (answered 99% when true, 0% when false) |  |
| RHQ | RHQ131 | Ever been pregnant? | all 3 | 53.9 | 0.0 | Females only 20 YEARS - 150 YEARS | subset: asked if RIAGENDR == 2 (answered 89% when true, 0% when false) |  |
| RHQ | RHQ160 | How many times have been pregnant? | all 3 | 62.0 | 0.0 | Females only 20 YEARS - 150 YEARS | subset: asked if RHQ131 == 1 (answered 100% when true, 0% when false) |  |
| RHQ | RHQ171 | How many deliveries live birth result? | all 3 | 64.1 | 0.0 | Females only 20 YEARS - 150 YEARS | subset: asked if RHQ131 == 1 (answered 94% when true, 0% when false) |  |
| RHQ | RHD180 | Age at first live birth | all 3 | 69.7 | 0.1 | Females only 20 YEARS - 150 YEARS | subset: asked if RHQ131 == 1 (answered 80% when true, 0% when false) |  |
| RHQ | RHQ172 | Any babies weigh 9 lbs or more? | all 3 | 64.1 | 0.2 | Females only 20 YEARS - 150 YEARS | subset: asked if RHQ131 == 1 (answered 94% when true, 0% when false) |  |
| RHQ | RHD280 | Had a hysterectomy? | all 3 | 53.9 | 0.0 | Females only 20 YEARS - 150 YEARS | subset: asked if RIAGENDR == 2 (answered 89% when true, 0% when false) |  |
| RHQ | RHQ540 | Ever use female hormones? | all 3 | 53.8 | 0.1 | Females only 20 YEARS - 150 YEARS | subset: asked if RIAGENDR == 2 (answered 89% when true, 0% when false) |  |
| RHQ | RHQ162 | During pregnancy, told you have diabetes | all 3 | 62.0 | 0.1 | Females only 20 YEARS - 150 YEARS | subset: asked if RHQ131 == 1 (answered 100% when true, 0% when false) | ⚠ gestational diabetes is an ADA risk item, but it is a past glucose-based diabetes diagnosis; women who later developed diabetes may recall it more (borderline) |
| DEMO | RIDEXPRG | Pregnancy status at exam | all 3 | 75.2 | 0.0 | Females only 20 YEARS - 44 YEARS | subset or non-response: 52% missing among eligible; gate not checked | ⚠ pregnancy at exam: HbA1c is lower in pregnancy; candidate EXCLUSION, not a feature |


### other: body measures and fitness

| file | variable | meaning | cycles | missing_pct | refused_dk_pct | target | universe | flag |
|---|---|---|---|---|---|---|---|---|
| BMX | BMXWT | Weight (kg) | all 3 | 0.9 | 0.0 | Both males and females 0 YEARS - 150 YEARS | everyone in target |  |
| BMX | BMXARMC | Arm Circumference (cm) | all 3 | 3.9 | 0.0 | Both males and females 2 MONTHS - 150 YEARS | everyone in target |  |
| BMX | BMXARML | Upper Arm Length (cm) | all 3 | 3.9 | 0.0 | Both males and females 2 MONTHS - 150 YEARS | everyone in target |  |
| BMX | BMXLEG | Upper Leg Length (cm) | all 3 | 4.9 | 0.0 | Both males and females 8 YEARS - 150 YEARS | everyone in target |  |
| BMX | BMDAVSAD | Average Sagittal Abdominal Diameter (cm) | all 3 | 7.5 | 0.0 | Both males and females 8 YEARS - 150 YEARS | subset or non-response: 7% missing among eligible; gate not checked |  |
| MGX | MGDCGSZ | Combined grip strength (kg) | 11-12, 13-14 | 9.2 | 0.0 | Both males and females 6 YEARS - 150 YEARS | subset or non-response: 9% missing among eligible; gate not checked |  |


### other: general and mental health

| file | variable | meaning | cycles | missing_pct | refused_dk_pct | target | universe | flag |
|---|---|---|---|---|---|---|---|---|
| HUQ | HUQ010 | General health condition | all 3 | 0.0 | 0.1 | Both males and females 0 YEARS - 150 YEARS | everyone in target |  |
| DPQ | DPQ010 | Have little interest in doing things | all 3 | 9.9 | 0.1 | Both males and females 18 YEARS - 150 YEARS | subset or non-response: 10% missing among eligible; gate not checked |  |
| DPQ | DPQ020 | Feeling down, depressed, or hopeless | all 3 | 9.9 | 0.1 | Both males and females 18 YEARS - 150 YEARS | subset or non-response: 10% missing among eligible; gate not checked |  |
| DPQ | DPQ030 | Trouble sleeping or sleeping too much | all 3 | 9.9 | 0.0 | Both males and females 18 YEARS - 150 YEARS | subset or non-response: 10% missing among eligible; gate not checked |  |
| DPQ | DPQ040 | Feeling tired or having little energy | all 3 | 10.0 | 0.0 | Both males and females 18 YEARS - 150 YEARS | subset or non-response: 10% missing among eligible; gate not checked |  |
| DPQ | DPQ050 | Poor appetite or overeating | all 3 | 10.0 | 0.0 | Both males and females 18 YEARS - 150 YEARS | subset or non-response: 10% missing among eligible; gate not checked |  |
| DPQ | DPQ060 | Feeling bad about yourself | all 3 | 10.0 | 0.0 | Both males and females 18 YEARS - 150 YEARS | subset or non-response: 10% missing among eligible; gate not checked |  |
| DPQ | DPQ070 | Trouble concentrating on things | all 3 | 10.0 | 0.0 | Both males and females 18 YEARS - 150 YEARS | subset or non-response: 10% missing among eligible; gate not checked |  |
| DPQ | DPQ080 | Moving or speaking slowly or too fast | all 3 | 10.0 | 0.0 | Both males and females 18 YEARS - 150 YEARS | subset or non-response: 10% missing among eligible; gate not checked |  |
| DPQ | DPQ090 | Thought you would be better off dead | all 3 | 10.0 | 0.0 | Both males and females 18 YEARS - 150 YEARS | subset or non-response: 10% missing among eligible; gate not checked |  |
| CDQ | CDQ001 | SP ever had pain or discomfort in chest | all 3 | 38.7 | 0.1 | Both males and females 40 YEARS - 150 YEARS | subset or non-response: 39% missing among eligible; gate not checked |  |
| CDQ | CDQ010 | Shortness of breath on stairs/inclines | all 3 | 38.7 | 0.1 | Both males and females 40 YEARS - 150 YEARS | subset or non-response: 39% missing among eligible; gate not checked |  |


### other: healthcare access

| file | variable | meaning | cycles | missing_pct | refused_dk_pct | target | universe | flag |
|---|---|---|---|---|---|---|---|---|
| HUQ | HUQ030 | Routine place to go for healthcare | all 3 | 0.0 | 0.0 | Both males and females 0 YEARS - 150 YEARS | everyone in target | ⚠ healthcare access: drives being diagnosed vs undiagnosed (selection into the label) |
| HUQ | HUQ051 | #times receive healthcare over past year | 13-14, 15-16 | 0.0 | 0.1 | Both males and females 0 YEARS - 150 YEARS | everyone in target | ⚠ as HUQ030 |
| HUQ | HUQ071 | Overnight hospital patient in last year | all 3 | 0.0 | 0.0 | Both males and females 0 YEARS - 150 YEARS | everyone in target | ⚠ as HUQ030 |


### other: comorbidities and advice

| file | variable | meaning | cycles | missing_pct | refused_dk_pct | target | universe | flag |
|---|---|---|---|---|---|---|---|---|
| MCQ | MCQ080 | Doctor ever said you were overweight | all 3 | 0.0 | 0.1 | Both males and females 16 YEARS - 150 YEARS | everyone in target | ⚠ doctor told overweight: follows a healthcare visit; mostly restates BMI |
| MCQ | MCQ300A | Close relative had heart attack? | all 3 | 0.0 | 2.3 | Both males and females 20 YEARS - 150 YEARS | everyone in target |  |
| MCQ | MCQ160B | Ever told had congestive heart failure | all 3 | 0.0 | 0.1 | Both males and females 20 YEARS - 150 YEARS | everyone in target | ⚠ cardiovascular disease is a diabetes complication (reverse causation) and implies healthcare contact |
| MCQ | MCQ160C | Ever told you had coronary heart disease | all 3 | 0.0 | 0.2 | Both males and females 20 YEARS - 150 YEARS | everyone in target | ⚠ cardiovascular disease is a diabetes complication (reverse causation) and implies healthcare contact |
| MCQ | MCQ160E | Ever told you had heart attack | all 3 | 0.0 | 0.1 | Both males and females 20 YEARS - 150 YEARS | everyone in target | ⚠ cardiovascular disease is a diabetes complication (reverse causation) and implies healthcare contact |
| MCQ | MCQ160F | Ever told you had a stroke | all 3 | 0.0 | 0.1 | Both males and females 20 YEARS - 150 YEARS | everyone in target | ⚠ cardiovascular disease is a diabetes complication (reverse causation) and implies healthcare contact |
| MCQ | MCQ160L | Ever told you had any liver condition | all 3 | 0.0 | 0.1 | Both males and females 20 YEARS - 150 YEARS | everyone in target | ⚠ liver disease (NAFLD link); diagnosis |
| MCQ | MCQ220 | Ever told you had cancer or malignancy | all 3 | 0.0 | 0.0 | Both males and females 20 YEARS - 150 YEARS | everyone in target |  |
| MCQ | MCQ365A | Doctor told you to lose weight | all 3 | 0.0 | 0.0 | Both males and females 16 YEARS - 150 YEARS | everyone in target | ⚠ doctor advice (lose weight / exercise / salt / fat) can follow abnormal glucose or HbA1c results: possible label leakage |
| MCQ | MCQ365B | Doctor told you to exercise | all 3 | 0.0 | 0.0 | Both males and females 16 YEARS - 150 YEARS | everyone in target | ⚠ doctor advice (lose weight / exercise / salt / fat) can follow abnormal glucose or HbA1c results: possible label leakage |
| MCQ | MCQ365C | Doctor told you to reduce salt in diet | all 3 | 0.0 | 0.1 | Both males and females 16 YEARS - 150 YEARS | everyone in target | ⚠ doctor advice (lose weight / exercise / salt / fat) can follow abnormal glucose or HbA1c results: possible label leakage |
| MCQ | MCQ365D | Doctor told you to reduce fat/calories | all 3 | 0.0 | 0.1 | Both males and females 16 YEARS - 150 YEARS | everyone in target | ⚠ doctor advice (lose weight / exercise / salt / fat) can follow abnormal glucose or HbA1c results: possible label leakage |
| MCQ | MCQ370A | Are you now controlling or losing weight | all 3 | 0.0 | 0.1 | Both males and females 16 YEARS - 150 YEARS | everyone in target | ⚠ behaviour change after the advice above: same leakage risk |
| MCQ | MCQ370B | Are you now increasing exercise | all 3 | 0.0 | 0.0 | Both males and females 16 YEARS - 150 YEARS | everyone in target | ⚠ behaviour change after the advice above: same leakage risk |
| MCQ | MCQ370C | Are you now reducing salt in diet | all 3 | 0.0 | 0.0 | Both males and females 16 YEARS - 150 YEARS | everyone in target | ⚠ behaviour change after the advice above: same leakage risk |
| MCQ | MCQ370D | Are you now reducing fat in diet | all 3 | 0.0 | 0.1 | Both males and females 16 YEARS - 150 YEARS | everyone in target | ⚠ behaviour change after the advice above: same leakage risk |
| KIQ_U | KIQ022 | Ever told you had weak/failing kidneys | all 3 | 0.0 | 0.1 | Both males and females 20 YEARS - 150 YEARS | everyone in target | ⚠ kidney disease is a diabetes complication (reverse causation) |
| KIQ_U | KIQ480 | How many times urinate in night? | all 3 | 9.6 | 0.0 | Both males and females 20 YEARS - 150 YEARS | subset or non-response: 10% missing among eligible; gate not checked | ⚠ nocturia: a symptom of hyperglycaemia (legitimate screening symptom, but close to the outcome) |


### other: diabetes-adjacent questions

| file | variable | meaning | cycles | missing_pct | refused_dk_pct | target | universe | flag |
|---|---|---|---|---|---|---|---|---|
| DIQ | DIQ160 | Ever told you have prediabetes | all 3 | 2.4 | 0.1 | Both males and females 12 YEARS - 150 YEARS | everyone in target | ⚠ PREDIABETES DIAGNOSIS: based on a glucose/HbA1c test: near-direct proxy for the label |
| DIQ | DIQ170 | Ever told have health risk for diabetes | all 3 | 0.0 | 0.2 | Both males and females 12 YEARS - 150 YEARS | everyone in target | ⚠ told at risk for diabetes: follows testing or a clinical judgement: label-adjacent |
| DIQ | DIQ172 | Feel could be at risk for diabetes | all 3 | 0.0 | 1.3 | Both males and females 12 YEARS - 150 YEARS | everyone in target | ⚠ feels at risk: may reflect known test results: label-adjacent |
| DIQ | DIQ180 | Had blood tested past three years | all 3 | 0.0 | 2.8 | Both males and females 12 YEARS - 150 YEARS | everyone in target | ⚠ had a blood test for diabetes in past 3 years: healthcare contact, and a negative test makes undiagnosed diabetes less likely |


## B. HbA1c by cycle, main-label sample rules applied (unweighted and WTMEC2YR-weighted)

| cycle | file | variable | label | target | pills_variable | analysis_n | positives | prevalence_unweighted_pct | prevalence_weighted_pct |
|---|---|---|---|---|---|---|---|---|---|
| 2005-2006 | GHB_D | LBXGH | Glycohemoglobin (%) | Both males and females 12 YEARS - 150 YEARS | DID070 | 4025 | 97 | 2.4 | 1.6 |
| 2007-2008 | GHB_E | LBXGH | Glycohemoglobin (%) | Both males and females 12 YEARS - 150 YEARS | DID070 | 4639 | 173 | 3.7 | 2.5 |
| 2009-2010 | GHB_F | LBXGH | Glycohemoglobin (%) | Both males and females 12 YEARS - 150 YEARS | DIQ070 | 5035 | 171 | 3.4 | 2.4 |
| 2011-2012 | GHB_G | LBXGH | Glycohemoglobin (%) | Both males and females 12 YEARS - 150 YEARS | DIQ070 | 4338 | 152 | 3.5 | 2.2 |
| 2013-2014 | GHB_H | LBXGH | Glycohemoglobin (%) | Both males and females 12 YEARS - 150 YEARS | DIQ070 | 4666 | 132 | 2.8 | 1.9 |
| 2015-2016 | GHB_I | LBXGH | Glycohemoglobin (%) | Both males and females 12 YEARS - 150 YEARS | DIQ070 | 4351 | 127 | 2.9 | 2.0 |


## C. Variable availability 2005-2016 (yes = present, - = absent, no file = component not fielded)

| file | variable | 2005-2006 | 2007-2008 | 2009-2010 | 2011-2012 | 2013-2014 | 2015-2016 |
|---|---|---|---|---|---|---|---|
| PAQ | PAQ605 | - | yes | yes | yes | yes | yes |
| PAQ | PAQ610 | - | yes | yes | yes | yes | yes |
| PAQ | PAD615 | - | yes | yes | yes | yes | yes |
| PAQ | PAQ620 | - | yes | yes | yes | yes | yes |
| PAQ | PAQ625 | - | yes | yes | yes | yes | yes |
| PAQ | PAD630 | - | yes | yes | yes | yes | yes |
| PAQ | PAQ635 | - | yes | yes | yes | yes | yes |
| PAQ | PAQ640 | - | yes | yes | yes | yes | yes |
| PAQ | PAD645 | - | yes | yes | yes | yes | yes |
| PAQ | PAQ650 | - | yes | yes | yes | yes | yes |
| PAQ | PAQ655 | - | yes | yes | yes | yes | yes |
| PAQ | PAD660 | - | yes | yes | yes | yes | yes |
| PAQ | PAQ665 | - | yes | yes | yes | yes | yes |
| PAQ | PAQ670 | - | yes | yes | yes | yes | yes |
| PAQ | PAD675 | - | yes | yes | yes | yes | yes |
| PAQ | PAD680 | - | yes | yes | yes | yes | yes |
| PAQ | PAQ710 | - | - | - | yes | yes | yes |
| PAQ | PAQ715 | - | - | - | yes | yes | yes |
| PAQ | PAD020 | yes | - | - | - | - | - |
| PAQ | PAQ100 | yes | - | - | - | - | - |
| PAQ | PAD200 | yes | - | - | - | - | - |
| PAQ | PAD320 | yes | - | - | - | - | - |
| SLQ | SLD010H | yes | yes | yes | yes | yes | - |
| SLQ | SLD012 | - | - | - | - | - | yes |
| SLQ | SLQ030 | yes | yes | - | - | - | yes |
| SLQ | SLQ040 | yes | yes | - | - | - | yes |
| SLQ | SLQ050 | yes | yes | yes | yes | yes | yes |
| SLQ | SLQ060 | yes | yes | yes | yes | yes | - |
| SLQ | SLQ120 | yes | yes | - | - | - | yes |
| SMQ | SMQ020 | yes | yes | yes | yes | yes | yes |
| SMQ | SMQ040 | yes | yes | yes | yes | yes | yes |
| SMQ | SMD030 | yes | yes | yes | yes | yes | yes |
| SMQ | SMD641 | yes | yes | yes | yes | yes | yes |
| SMQ | SMD650 | yes | yes | yes | yes | yes | yes |
| SMQ | SMQ050Q | yes | yes | yes | yes | yes | yes |
| SMQ | SMQ050U | yes | yes | yes | yes | yes | yes |
| ALQ | ALQ101 | yes | yes | yes | yes | yes | yes |
| ALQ | ALQ110 | yes | yes | yes | yes | yes | yes |
| ALQ | ALQ120Q | yes | yes | yes | yes | yes | yes |
| ALQ | ALQ120U | yes | yes | yes | yes | yes | yes |
| ALQ | ALQ130 | yes | yes | yes | yes | yes | yes |
| ALQ | ALQ141Q | - | - | - | yes | yes | yes |
| ALQ | ALQ141U | - | - | - | yes | yes | yes |
| ALQ | ALQ151 | - | - | - | yes | yes | yes |
| ALQ | ALQ140Q | yes | yes | yes | - | - | - |
| DBQ | DBQ700 | yes | yes | yes | yes | yes | yes |
| DBQ | DBD895 | - | yes | yes | yes | yes | yes |
| DBQ | DBD900 | - | yes | yes | yes | yes | yes |
| DBQ | DBD905 | - | yes | yes | yes | yes | yes |
| DBQ | DBD910 | - | yes | yes | yes | yes | yes |
| DBQ | DBQ197 | yes | yes | yes | yes | yes | yes |
| DR1TOT | DR1TKCAL | yes | yes | yes | yes | yes | yes |
| DR1TOT | DR1TCARB | yes | yes | yes | yes | yes | yes |
| DR1TOT | DR1TSUGR | yes | yes | yes | yes | yes | yes |
| DR1TOT | DR1TFIBE | yes | yes | yes | yes | yes | yes |
| DR1TOT | DR1TTFAT | yes | yes | yes | yes | yes | yes |
| DR1TOT | DR1TSFAT | yes | yes | yes | yes | yes | yes |
| DR1TOT | DR1TSODI | yes | yes | yes | yes | yes | yes |
| DR1TOT | DR1TALCO | yes | yes | yes | yes | yes | yes |
| DR1TOT | DR1DRSTZ | yes | yes | yes | yes | yes | yes |
| DEMO | DMDEDUC2 | yes | yes | yes | yes | yes | yes |
| DEMO | INDFMPIR | yes | yes | yes | yes | yes | yes |
| DEMO | INDHHIN2 | - | yes | yes | yes | yes | yes |
| DEMO | DMDMARTL | yes | yes | yes | yes | yes | yes |
| DEMO | DMDHHSIZ | yes | yes | yes | yes | yes | yes |
| DEMO | DMDBORN4 | - | - | - | yes | yes | yes |
| DEMO | DMDCITZN | yes | yes | yes | yes | yes | yes |
| DEMO | DMDYRSUS | yes | yes | yes | yes | yes | yes |
| DEMO | SIALANG | yes | yes | yes | yes | yes | yes |
| DEMO | RIDEXPRG | yes | yes | yes | yes | yes | yes |
| DEMO | RIAGENDR | yes | yes | yes | yes | yes | yes |
| DEMO | RIDAGEYR | yes | yes | yes | yes | yes | yes |
| DEMO | RIDRETH1 | yes | yes | yes | yes | yes | yes |
| DEMO | RIDRETH3 | - | - | - | yes | yes | yes |
| DEMO | WTMEC2YR | yes | yes | yes | yes | yes | yes |
| DEMO | DMDBORN | yes | - | - | - | - | - |
| DEMO | DMDBORN2 | - | yes | yes | - | - | - |
| DEMO | INDHHINC | yes | - | - | - | - | - |
| FSQ | FSDHH | yes | yes | yes | yes | yes | yes |
| FSQ | FSDAD | yes | yes | yes | yes | yes | yes |
| FSQ | FSQ165 | - | yes | yes | yes | yes | yes |
| HIQ | HIQ011 | yes | yes | yes | yes | yes | yes |
| OCQ | OCD150 | yes | yes | yes | yes | yes | yes |
| OCQ | OCQ180 | yes | yes | yes | yes | yes | yes |
| WHQ | WHD010 | yes | yes | yes | yes | yes | yes |
| WHQ | WHD020 | yes | yes | yes | yes | yes | yes |
| WHQ | WHQ030 | yes | yes | yes | yes | yes | yes |
| WHQ | WHQ040 | yes | yes | yes | yes | yes | yes |
| WHQ | WHD050 | yes | yes | yes | yes | yes | yes |
| WHQ | WHD110 | yes | yes | yes | yes | yes | yes |
| WHQ | WHD120 | yes | yes | yes | yes | yes | yes |
| WHQ | WHD130 | yes | yes | yes | yes | yes | yes |
| WHQ | WHD140 | yes | yes | yes | yes | yes | yes |
| WHQ | WHQ150 | yes | yes | yes | yes | yes | yes |
| WHQ | WHQ070 | yes | yes | yes | yes | yes | yes |
| WHQ | WHQ060 | yes | yes | yes | yes | yes | yes |
| BPQ | BPQ020 | yes | yes | yes | yes | yes | yes |
| BPQ | BPQ030 | yes | yes | yes | yes | yes | yes |
| BPQ | BPD035 | - | yes | yes | yes | yes | yes |
| BPQ | BPQ040A | yes | yes | yes | yes | yes | yes |
| BPQ | BPQ050A | yes | yes | yes | yes | yes | yes |
| BPQ | BPQ060 | yes | yes | yes | yes | yes | yes |
| BPQ | BPQ080 | yes | yes | yes | yes | yes | yes |
| BPQ | BPQ090D | yes | yes | yes | yes | yes | yes |
| BPX | BPXPLS | yes | yes | yes | yes | yes | yes |
| BPX | BPXSY1 | yes | yes | yes | yes | yes | yes |
| BPX | BPXDI1 | yes | yes | yes | yes | yes | yes |
| RHQ | RHQ010 | yes | yes | yes | yes | yes | yes |
| RHQ | RHQ031 | yes | yes | yes | yes | yes | yes |
| RHQ | RHQ060 | yes | yes | yes | yes | yes | yes |
| RHQ | RHQ131 | yes | yes | yes | yes | yes | yes |
| RHQ | RHQ160 | yes | yes | yes | yes | yes | yes |
| RHQ | RHQ171 | yes | yes | yes | yes | yes | yes |
| RHQ | RHD180 | - | yes | yes | yes | yes | yes |
| RHQ | RHQ172 | - | yes | yes | yes | yes | yes |
| RHQ | RHD280 | yes | yes | yes | yes | yes | yes |
| RHQ | RHQ540 | yes | yes | yes | yes | yes | yes |
| RHQ | RHQ162 | - | yes | yes | yes | yes | yes |
| BMX | BMXWT | yes | yes | yes | yes | yes | yes |
| BMX | BMXARMC | yes | yes | yes | yes | yes | yes |
| BMX | BMXARML | yes | yes | yes | yes | yes | yes |
| BMX | BMXLEG | yes | yes | yes | yes | yes | yes |
| BMX | BMDAVSAD | - | - | - | yes | yes | yes |
| BMX | BMXBMI | yes | yes | yes | yes | yes | yes |
| BMX | BMXWAIST | yes | yes | yes | yes | yes | yes |
| BMX | BMXHT | yes | yes | yes | yes | yes | yes |
| MGX | MGDCGSZ | no file | no file | no file | yes | yes | no file |
| HUQ | HUQ010 | yes | yes | yes | yes | yes | yes |
| HUQ | HUQ030 | yes | yes | yes | yes | yes | yes |
| HUQ | HUQ051 | - | - | - | - | yes | yes |
| HUQ | HUQ071 | yes | yes | yes | yes | yes | yes |
| HUQ | HUQ050 | yes | yes | yes | yes | - | - |
| DPQ | DPQ010 | yes | yes | yes | yes | yes | yes |
| DPQ | DPQ020 | yes | yes | yes | yes | yes | yes |
| DPQ | DPQ030 | yes | yes | yes | yes | yes | yes |
| DPQ | DPQ040 | yes | yes | yes | yes | yes | yes |
| DPQ | DPQ050 | yes | yes | yes | yes | yes | yes |
| DPQ | DPQ060 | yes | yes | yes | yes | yes | yes |
| DPQ | DPQ070 | yes | yes | yes | yes | yes | yes |
| DPQ | DPQ080 | yes | yes | yes | yes | yes | yes |
| DPQ | DPQ090 | yes | yes | yes | yes | yes | yes |
| CDQ | CDQ001 | yes | yes | yes | yes | yes | yes |
| CDQ | CDQ010 | yes | yes | yes | yes | yes | yes |
| MCQ | MCQ080 | yes | yes | yes | yes | yes | yes |
| MCQ | MCQ300A | yes | yes | yes | yes | yes | yes |
| MCQ | MCQ160B | yes | yes | yes | yes | yes | yes |
| MCQ | MCQ160C | yes | yes | yes | yes | yes | yes |
| MCQ | MCQ160E | yes | yes | yes | yes | yes | yes |
| MCQ | MCQ160F | yes | yes | yes | yes | yes | yes |
| MCQ | MCQ160L | yes | yes | yes | yes | yes | yes |
| MCQ | MCQ220 | yes | yes | yes | yes | yes | yes |
| MCQ | MCQ365A | - | - | - | yes | yes | yes |
| MCQ | MCQ365B | - | - | - | yes | yes | yes |
| MCQ | MCQ365C | - | - | - | yes | yes | yes |
| MCQ | MCQ365D | - | - | - | yes | yes | yes |
| MCQ | MCQ370A | - | - | - | yes | yes | yes |
| MCQ | MCQ370B | - | - | - | yes | yes | yes |
| MCQ | MCQ370C | - | - | - | yes | yes | yes |
| MCQ | MCQ370D | - | - | - | yes | yes | yes |
| MCQ | MCQ300C | yes | yes | yes | yes | yes | yes |
| KIQ_U | KIQ022 | yes | yes | yes | yes | yes | yes |
| KIQ_U | KIQ480 | yes | yes | yes | yes | yes | yes |
| DIQ | DIQ160 | yes | yes | yes | yes | yes | yes |
| DIQ | DIQ170 | yes | yes | yes | yes | yes | yes |
| DIQ | DIQ172 | - | - | - | yes | yes | yes |
| DIQ | DIQ180 | yes | yes | yes | yes | yes | yes |
| DIQ | DIQ010 | yes | yes | yes | yes | yes | yes |
| DIQ | DIQ050 | yes | yes | yes | yes | yes | yes |
| DIQ | DIQ070 | - | - | yes | yes | yes | yes |
| GHB | LBXGH | yes | yes | yes | yes | yes | yes |
| GLU | LBXGLU | yes | yes | yes | yes | yes | yes |
| GLU | WTSAF2YR | yes | yes | yes | yes | yes | yes |

<!-- GENERATED TABLES END -->
