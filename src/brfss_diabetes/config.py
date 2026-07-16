from __future__ import annotations

from pathlib import Path

PROJECT_TITLE = "Diabetes Risk Prediction using BRFSS Health Indicators"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 3
MODEL_COMPARISON_SAMPLE = 30_000
TUNING_SAMPLE = 25_000
ABLATION_TRAIN_SAMPLE = 30_000
ABLATION_VALIDATION_SAMPLE = 10_000

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "diabetes_012_health_indicators_BRFSS2015.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "brfss_final"
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"
MODEL_DIR = PROJECT_ROOT / "models" / "brfss_final"

TARGET = "Diabetes_012"
ORIGINAL_FEATURES = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]

BINARY_FEATURES = {
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
    "Sex",
}
ORDINAL_FEATURES = {"GenHlth", "Age", "Education", "Income"}
CONTINUOUS_FEATURES = {"BMI", "MentHlth", "PhysHlth"}
ENGINEERED_CATEGORICAL = {"bmi_category", "age_band"}

CLASS_NAMES = {
    0: "No diabetes",
    1: "Prediabetes",
    2: "Diabetes",
}


def ensure_directories() -> None:
    for path in (
        PROCESSED_DATA_DIR,
        TABLE_DIR,
        FIGURE_DIR,
        MODEL_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
