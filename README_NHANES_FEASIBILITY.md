# Pre-Test Diabetes Risk Prioritisation using Non-Glycemic NHANES Indicators

This directory contains a feasibility pilot. It does **not** replace the existing BRFSS project unless the held-out acceptance criteria are met.

The model does not diagnose diabetes. It ranks adults by estimated likelihood of belonging to a high-risk group so that limited confirmatory HbA1c, fasting-glucose or OGTT capacity can be prioritised.

Glycemic biomarkers and self-reported diabetes status are used only to construct the ground-truth label. They are excluded from all model predictors.

Last verified by full local execution: **2026-07-24**, using the pinned versions in `requirements.txt`.

## Current verified decision

**NO-GO — retain BRFSS as the official project.** The refreshed XGBoost high-risk PR-AUC is **0.809**. Testing the top 40% captures **59.1%** of high-risk participants; reaching 80% capture requires testing **60.9%** of participants. None of the predefined acceptance criteria passes.

## Data source

The pipeline uses official NHANES 2015–2016 public SAS XPORT (`.XPT`) files from CDC/NCHS. The project brief's legacy URL is attempted first. Because that endpoint currently returns a CDC Page Not Found HTML response, the downloader validates the XPT header and falls back to the current official path:

`https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/<FILENAME>`

No Kaggle, UCI or Pima dataset is used.

## Windows Git Bash setup

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/run_nhanes_feasibility.py
```

## PowerShell setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/run_nhanes_feasibility.py
```

Optional file check:

```powershell
python scripts/check_nhanes_files.py
```

## Target

`diabetes_risk_stage` is assigned using the highest-risk available evidence:

- Class 2: doctor-diagnosed diabetes, HbA1c at least 6.5%, fasting glucose at least 126 mg/dL, or 2-hour OGTT at least 200 mg/dL.
- Class 1: self-reported borderline diabetes, HbA1c 5.7–6.4%, fasting glucose 100–125 mg/dL, or 2-hour OGTT 140–199 mg/dL.
- Class 0: no available class 1 or class 2 criterion is met.

Adults with no usable self-report or biomarker label evidence are excluded. Class 0 is therefore a lower-risk label under available evidence, not proof that dysglycemia is absent.

## Main predictors

Only available Tier-1 non-glycemic variables are used: demographics, anthropometrics, blood pressure, non-diabetes medical history, lifestyle and socioeconomic/access indicators. The pipeline creates simple domain features such as average blood pressure, waist-to-height ratio, central obesity, age/BMI bands and cardiometabolic history counts.

The pipeline logs and skips missing files or variables. It generates file, variable and missingness reports under `reports/nhanes_feasibility/tables/`.

## Models and decision test

The pilot compares:

- class-weighted logistic regression;
- lightly tuned multiclass XGBoost;
- a transparent points rule based on age, body size, blood pressure, hypertension, inactivity and smoking.

The main business test ranks held-out participants by `P(Class 1) + P(Class 2)` and simulates sending only the top 10%, 20%, 30%, 40% or 50% for confirmatory testing.

The final decision is written to:

- `reports/nhanes_feasibility/NHANES_PILOT_VERDICT.md`
- `reports/nhanes_feasibility/FEASIBILITY_SUMMARY.md`
- `reports/nhanes_feasibility/report.html`
- `notebooks/01_nhanes_feasibility_eda.ipynb`

If none of the predefined criteria passes, BRFSS remains the official project.
