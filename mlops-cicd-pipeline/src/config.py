"""
config.py

Central configuration file for the Credit Card Fraud Detection
MLOps Pipeline.

All project paths, model settings, preprocessing parameters,
and environment variables are managed from here.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# ============================================================
# Load Environment Variables
# ============================================================

load_dotenv()

# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

TEST_DIR = PROJECT_ROOT / "tests"

LOG_DIR = PROJECT_ROOT / "logs"

MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# ============================================================
# Dataset Paths
# ============================================================

RAW_DATA_PATH = RAW_DATA_DIR / "creditcard.csv"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"

TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

SCALER_PATH = MODEL_DIR / "scaler.joblib"

# ============================================================
# Model Configuration
# ============================================================

MODEL_NAME = "credit_card_fraud_model.joblib"

MODEL_PATH = MODEL_DIR / MODEL_NAME

TARGET_COLUMN = "Class"

# V1-V28 (PCA components) + Time + Amount. Fixed and known for this
# dataset, so we declare them explicitly rather than infer at runtime.
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]

# ============================================================
# Data Splitting
# ============================================================

TEST_SIZE = 0.20

VALIDATION_SIZE = 0.10

RANDOM_STATE = 42

STRATIFY = True

# ============================================================
# Preprocessing Configuration
# ============================================================

DROP_DUPLICATES = True

SCALE_AMOUNT = True

SCALE_TIME = False

SCALER_TYPE = "StandardScaler"

# ============================================================
# Training Configuration
# ============================================================

DEFAULT_MODEL = "RandomForest"

# class_weight="balanced" is required here, not optional: fraud is
# ~0.17% of this dataset, so an unweighted RandomForest will happily
# predict "not fraud" every time and still score ~99.8% accuracy
# while catching zero fraud cases.
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "solver": "liblinear",
    "class_weight": "balanced",
    "random_state": RANDOM_STATE
}

# ============================================================
# MLflow Configuration
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "Credit Card Fraud Detection"
)

# ============================================================
# API Configuration
# ============================================================

HOST = os.getenv("HOST", "0.0.0.0")

PORT = int(os.getenv("PORT", 8000))

DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# ============================================================
# Logging Configuration
# ============================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

LOG_FILE = LOG_DIR / "pipeline.log"

# ============================================================
# Evaluation Configuration
# ============================================================

# Accuracy is not meaningful on a ~0.17%-positive dataset, so CI/CD
# gates on recall (or f1) instead. Both are read from .env so the
# threshold can be tuned without touching code.
EVAL_METRIC = os.getenv("EVAL_METRIC", "recall")

MIN_RECALL_THRESHOLD = float(os.getenv("MIN_RECALL_THRESHOLD", 0.75))

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]

# ============================================================
# Create Required Directories
# ============================================================

REQUIRED_DIRECTORIES = [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    LOG_DIR,
    MLRUNS_DIR
]

for directory in REQUIRED_DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# Supported Models
# ============================================================

SUPPORTED_MODELS = [
    "LogisticRegression",
    "RandomForest"
]

# ============================================================
# Display Configuration (Optional)
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Credit Card Fraud Detection Configuration")
    print("=" * 60)

    print(f"Project Root       : {PROJECT_ROOT}")
    print(f"Dataset            : {RAW_DATA_PATH}")
    print(f"Processed Data     : {PROCESSED_DATA_DIR}")
    print(f"Model Directory    : {MODEL_DIR}")
    print(f"MLflow Directory   : {MLRUNS_DIR}")
    print(f"Target Column      : {TARGET_COLUMN}")
    print(f"Test Size          : {TEST_SIZE}")
    print(f"Random State       : {RANDOM_STATE}")
    print(f"Eval Metric        : {EVAL_METRIC}")
    print(f"Min Recall Thresh. : {MIN_RECALL_THRESHOLD}")
    print(f"API Host           : {HOST}")
    print(f"API Port           : {PORT}")
    print("=" * 60)