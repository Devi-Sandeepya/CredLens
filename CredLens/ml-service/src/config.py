from pathlib import Path
import os


# ============================================================
# PROJECT ROOTS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

ML_SERVICE = ROOT / "ml-service"
ARTIFACTS = ML_SERVICE / "artifacts"

FEATURES_DIR = ARTIFACTS / "features"
MODELS_DIR = ARTIFACTS / "models"
METRICS_DIR = ARTIFACTS / "metrics"


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

DATA_PROCESSED.mkdir(
    parents=True,
    exist_ok=True,
)

ARTIFACTS.mkdir(
    parents=True,
    exist_ok=True,
)

FEATURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATA SETTINGS
# ============================================================

SAMPLE_ROWS = int(
    os.getenv(
        "CREDLENS_SAMPLE_ROWS",
        "30000",
    )
)


# ============================================================
# P0 FEATURE TABLE
# ============================================================

APPLICANT_FEATURES_PATH = (
    FEATURES_DIR
    / "m5_features.parquet"
)


# ============================================================
# P0 RISK MODEL
# ============================================================

RISK_MODEL_PATH = (
    MODELS_DIR
    / "m5_risk_model.joblib"
)

RISK_FEATURES_PATH = (
    MODELS_DIR
    / "m5_risk_features.json"
)


# ============================================================
# P0 INTEGRITY MODEL
# ============================================================

INTEGRITY_MODEL_PATH = (
    MODELS_DIR
    / "integrity_model.joblib"
)

INTEGRITY_FEATURES_PATH = (
    MODELS_DIR
    / "integrity_features.json"
)


# ============================================================
# EXPERIMENT RESULTS
# ============================================================

ABLATION_RESULTS_PATH = (
    METRICS_DIR
    / "m1_m5_ablation.json"
)

ABLATION_RESULTS_CSV_PATH = (
    METRICS_DIR
    / "m1_m5_ablation.csv"
)