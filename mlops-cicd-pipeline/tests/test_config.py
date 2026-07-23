"""
test_config.py

Unit tests for src.config.

These tests verify that important configuration values,
paths, and model settings are defined correctly.
"""

from pathlib import Path

from src import config


# ============================================================
# Project Paths
# ============================================================

def test_project_root_exists():
    """PROJECT_ROOT should exist."""
    assert config.PROJECT_ROOT.exists()
    assert config.PROJECT_ROOT.is_dir()


def test_data_directory():
    """DATA_DIR should be a valid Path."""
    assert isinstance(config.DATA_DIR, Path)


def test_models_directory():
    """MODEL_DIR should be a valid Path."""
    assert isinstance(config.MODEL_DIR, Path)


def test_required_directories_exist():
    """
    config.py creates REQUIRED_DIRECTORIES on import (see the
    directory-creation loop at the bottom of the file) — verify that
    actually happened rather than assuming it did.
    """
    for directory in config.REQUIRED_DIRECTORIES:
        assert directory.exists()
        assert directory.is_dir()


# ============================================================
# Dataset Paths
# ============================================================

def test_raw_data_path():
    """RAW_DATA_PATH should point to a CSV file."""
    assert config.RAW_DATA_PATH.suffix == ".csv"


def test_processed_data_paths():
    """
    Processed train/test datasets should have CSV extensions.

    Note: config.py does not define a VAL_DATA_PATH constant — the
    validation split path is derived locally in preprocess.py/train.py
    as PROCESSED_DATA_DIR / "val.csv", by design, so it isn't tested
    here as a config attribute.
    """
    assert config.TRAIN_DATA_PATH.suffix == ".csv"
    assert config.TEST_DATA_PATH.suffix == ".csv"

    assert config.TRAIN_DATA_PATH.parent == config.PROCESSED_DATA_DIR
    assert config.TEST_DATA_PATH.parent == config.PROCESSED_DATA_DIR


def test_scaler_and_model_paths():
    """SCALER_PATH and MODEL_PATH should point to .joblib files under MODEL_DIR."""
    assert config.SCALER_PATH.suffix == ".joblib"
    assert config.MODEL_PATH.suffix == ".joblib"

    assert config.SCALER_PATH.parent == config.MODEL_DIR
    assert config.MODEL_PATH.parent == config.MODEL_DIR


# ============================================================
# Feature Configuration
# ============================================================

def test_target_column():
    """TARGET_COLUMN should be the fraud label column."""
    assert config.TARGET_COLUMN == "Class"


def test_feature_columns():
    """Verify feature list."""

    assert isinstance(config.FEATURE_COLUMNS, list)

    assert len(config.FEATURE_COLUMNS) == 30

    assert "Time" in config.FEATURE_COLUMNS
    assert "Amount" in config.FEATURE_COLUMNS

    for i in range(1, 29):
        assert f"V{i}" in config.FEATURE_COLUMNS


# ============================================================
# Data Splitting
# ============================================================

def test_split_sizes():
    """Test/validation split fractions should be sane proportions."""
    assert 0 < config.TEST_SIZE < 1
    assert 0 < config.VALIDATION_SIZE < 1
    assert config.TEST_SIZE + config.VALIDATION_SIZE < 1
    assert isinstance(config.STRATIFY, bool)


def test_random_state():
    """Random seed should be a non-negative integer."""

    assert isinstance(config.RANDOM_STATE, int)
    assert config.RANDOM_STATE >= 0


# ============================================================
# Preprocessing Configuration
# ============================================================

def test_scaling_flags():
    """Scaling flags should be booleans."""

    assert isinstance(config.SCALE_AMOUNT, bool)
    assert isinstance(config.SCALE_TIME, bool)


def test_scaler_type_supported():
    """SCALER_TYPE should be a recognized sklearn scaler name."""
    assert config.SCALER_TYPE in ("StandardScaler", "MinMaxScaler", "RobustScaler")


# ============================================================
# Model Parameters
# ============================================================

def test_random_forest_params():
    """Random Forest configuration."""

    params = config.RANDOM_FOREST_PARAMS

    assert isinstance(params, dict)

    assert params["random_state"] == config.RANDOM_STATE

    assert params["class_weight"] == "balanced"

    assert params["n_estimators"] > 0


def test_logistic_regression_params():
    """Logistic Regression configuration."""

    params = config.LOGISTIC_REGRESSION_PARAMS

    assert isinstance(params, dict)

    assert params["random_state"] == config.RANDOM_STATE

    assert params["class_weight"] == "balanced"


# ============================================================
# Supported Models
# ============================================================

def test_supported_models():
    """
    SUPPORTED_MODELS should list every model train.py/evaluate.py can
    build, and DEFAULT_MODEL should be one of them.
    """

    assert isinstance(config.SUPPORTED_MODELS, list)

    assert "RandomForest" in config.SUPPORTED_MODELS
    assert "LogisticRegression" in config.SUPPORTED_MODELS

    assert config.DEFAULT_MODEL in config.SUPPORTED_MODELS


# ============================================================
# Evaluation Configuration
# ============================================================

def test_recall_threshold():
    """Recall threshold should be between 0 and 1."""

    assert 0 <= config.MIN_RECALL_THRESHOLD <= 1


def test_eval_metric_is_valid():
    """EVAL_METRIC should be one of the metrics evaluate.py actually computes."""
    assert config.EVAL_METRIC in config.METRICS


def test_metrics_list():
    """METRICS should be a non-empty list of metric names."""
    assert isinstance(config.METRICS, list)
    assert len(config.METRICS) > 0
    assert "recall" in config.METRICS


# ============================================================
# MLflow Configuration
# ============================================================

def test_mlflow_config():
    """MLflow tracking URI and experiment name should be non-empty strings."""
    assert isinstance(config.MLFLOW_TRACKING_URI, str)
    assert len(config.MLFLOW_TRACKING_URI) > 0

    assert isinstance(config.MLFLOW_EXPERIMENT_NAME, str)
    assert len(config.MLFLOW_EXPERIMENT_NAME) > 0


# ============================================================
# API Configuration
# ============================================================

def test_api_config():
    """API configuration values."""

    assert isinstance(config.HOST, str)

    assert isinstance(config.PORT, int)

    assert 0 < config.PORT < 65536

    assert isinstance(config.DEBUG, bool)


# ============================================================
# Logging Configuration
# ============================================================

def test_logging_config():
    """LOG_FILE should live under LOG_DIR and use a .log extension."""
    assert config.LOG_FILE.parent == config.LOG_DIR
    assert config.LOG_FILE.suffix == ".log"
    assert isinstance(config.LOG_LEVEL, str)