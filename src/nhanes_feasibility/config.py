from __future__ import annotations

from pathlib import Path

PROJECT_NAME = "Pre-Test Diabetes Risk Prioritisation using Non-Glycemic NHANES Indicators"
CYCLE = "2015-2016"
RANDOM_STATE = 42
TEST_SIZE = 0.25

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "nhanes"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "nhanes_feasibility"
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"

# The first URL is retained because it was specified in the project brief. The
# CDC currently serves a Page Not Found HTML document at that path. Downloads
# therefore validate the XPT signature and fall back to the current official
# NHANES public-data path.
REQUESTED_BASE_URL = "https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016"
OFFICIAL_FALLBACK_BASE_URL = (
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles"
)

NHANES_FILES = {
    "demographics": "DEMO_I.XPT",
    "diabetes_questionnaire": "DIQ_I.XPT",
    "body_measures": "BMX_I.XPT",
    "blood_pressure_exam": "BPX_I.XPT",
    "blood_pressure_questionnaire": "BPQ_I.XPT",
    "medical_conditions": "MCQ_I.XPT",
    "physical_activity": "PAQ_I.XPT",
    "smoking": "SMQ_I.XPT",
    "alcohol": "ALQ_I.XPT",
    "diet": "DBQ_I.XPT",
    "health_insurance": "HIQ_I.XPT",
    "income": "INQ_I.XPT",
    "glycohemoglobin": "GHB_I.XPT",
    "fasting_glucose": "GLU_I.XPT",
    "ogtt": "OGTT_I.XPT",
}

LABEL_VARIABLES = {
    "self_reported_diabetes": "DIQ010",
    "self_reported_prediabetes": "DIQ160",
    "glycohemoglobin_percent": "LBXGH",
    "fasting_glucose_mg_dl": "LBXGLU",
    "ogtt_2h_glucose_mg_dl": "LBXGLT",
}

LEAKAGE_VARIABLES = {
    "DIQ010",
    "DIQ160",
    "LBXGH",
    "LBXGLU",
    "LBXGLT",
    "LBDGLUSI",
    "LBDGLTSI",
}

TOP_K_VALUES = (0.10, 0.20, 0.30, 0.40, 0.50)
CAPTURE_TARGETS = (0.70, 0.80, 0.90)


def ensure_directories() -> None:
    for path in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        TABLE_DIR,
        FIGURE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
