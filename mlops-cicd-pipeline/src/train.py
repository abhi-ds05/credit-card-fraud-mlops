"""
train.py

Model training pipeline for the Credit Card Fraud Detection
MLOps project.

Responsibilities:
- Load processed train / validation splits (produced by preprocess.py)
- Build a model (RandomForest or LogisticRegression) from config.py params
- Fit the model on the training split
- Evaluate on the validation split
- Log params/metrics/model to MLflow
- Persist the trained model to models/ via utils.save_object

This script is meant to be run standalone:
    python src/train.py
    python src/train.py --model LogisticRegression
"""

import argparse
import sys

import mlflow  # type: ignore[import-not-found]
import mlflow.sklearn  # type: ignore[import-not-found]
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    TRAIN_DATA_PATH,
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    MODEL_PATH,
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    RANDOM_FOREST_PARAMS,
    LOGISTIC_REGRESSION_PARAMS,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    RANDOM_STATE,
)
from src.utils import logger, save_object, set_random_seed, print_metrics

# preprocess.py writes val.csv here but config.py doesn't expose a
# constant for it (only TRAIN_DATA_PATH / TEST_DATA_PATH) — derive it
# the same way preprocess.py does, so both scripts stay in sync.
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val.csv"

MODEL_REGISTRY = {
    "RandomForest": (RandomForestClassifier, RANDOM_FOREST_PARAMS),
    "LogisticRegression": (LogisticRegression, LOGISTIC_REGRESSION_PARAMS),
}


# ============================================================
# Load
# ============================================================

def load_split(path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load a processed CSV split and separate features from the target.
    """
    if not path.exists():
        logger.error("Processed data file not found at %s", path)
        raise FileNotFoundError(
            f"{path} not found. Run `python src/preprocess.py` first."
        )

    df = pd.read_csv(path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    logger.info("Loaded %s with shape %s", path, df.shape)

    return X, y


# ============================================================
# Model
# ============================================================

def build_model(model_name: str):
    """
    Instantiate a model from MODEL_REGISTRY using params defined in config.py.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported options: {SUPPORTED_MODELS}"
        )

    model_cls, params = MODEL_REGISTRY[model_name]
    logger.info("Building %s with params: %s", model_name, params)

    return model_cls(**params), params


# ============================================================
# Train
# ============================================================

def train_model(model, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Fit the model on the training split.
    """
    logger.info("Training model on %d samples, %d features", *X_train.shape)
    model.fit(X_train, y_train)
    logger.info("Model training complete.")

    return model


# ============================================================
# Evaluate (on validation split, for model-selection purposes)
# ============================================================

def evaluate_model(model, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """
    Compute validation metrics. This is a lightweight check used to
    sanity-test the freshly trained model before handing off to
    evaluate.py, which runs the authoritative evaluation against the
    held-out test set for CI/CD gating.
    """
    y_pred = model.predict(X_val)

    # Not every model/config combination exposes predict_proba
    # (e.g. some SVM configurations), so guard roc_auc separately.
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_scores)
    else:
        roc_auc = float("nan")

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "roc_auc": roc_auc,
    }

    logger.info("Validation metrics: %s", metrics)
    print_metrics(metrics)

    return metrics


# ============================================================
# MLflow
# ============================================================

def log_to_mlflow(model, model_name: str, params: dict, metrics: dict) -> None:
    """
    Log the run to MLflow: params, validation metrics, and the model
    artifact. Falls back to a warning (rather than crashing the whole
    training run) if the tracking server is unreachable, since local/CI
    training shouldn't hard-fail just because MLflow isn't up.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run():
            mlflow.set_tag("model_name", model_name)
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info("Run logged to MLflow at %s", MLFLOW_TRACKING_URI)
    except Exception as exc:
        logger.warning("Could not log run to MLflow (%s). Continuing without it.", exc)


# ============================================================
# Orchestration
# ============================================================

def run_training(model_name: str = DEFAULT_MODEL) -> None:
    """
    Run the full training pipeline end to end.
    """
    set_random_seed(RANDOM_STATE)

    X_train, y_train = load_split(TRAIN_DATA_PATH)
    X_val, y_val = load_split(VAL_DATA_PATH)

    model, params = build_model(model_name)

    model = train_model(model, X_train, y_train)

    metrics = evaluate_model(model, X_val, y_val)

    log_to_mlflow(model, model_name, params, metrics)

    save_object(model, MODEL_PATH)

    logger.info("Training pipeline completed successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fraud detection model.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=SUPPORTED_MODELS,
        help=f"Model to train. Options: {SUPPORTED_MODELS} (default: {DEFAULT_MODEL})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        run_training(model_name=args.model)
    except Exception as exc:
        logger.exception("Training pipeline failed: %s", exc)
        sys.exit(1)