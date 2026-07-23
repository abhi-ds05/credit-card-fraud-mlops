"""
evaluate.py

Model evaluation pipeline for the Credit Card Fraud Detection
MLOps project.

Responsibilities:
- Load the held-out test split (produced by preprocess.py)
- Load the trained model (produced by train.py)
- Compute evaluation metrics on the test set
- Log the evaluation run to MLflow
- Gate CI/CD: exit non-zero if the configured metric falls below
  MIN_RECALL_THRESHOLD (or the equivalent for EVAL_METRIC)

This script is meant to be run standalone, after preprocess.py and
train.py:
    python src/evaluate.py
"""

import json
import sys

import mlflow # type: ignore
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    TEST_DATA_PATH,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    MODEL_PATH,
    MODEL_DIR,
    EVAL_METRIC,
    MIN_RECALL_THRESHOLD,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)
from src.utils import logger, load_object, print_metrics

METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"


# ============================================================
# Load
# ============================================================

def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the held-out test split.
    """
    if not TEST_DATA_PATH.exists():
        logger.error("Test data file not found at %s", TEST_DATA_PATH)
        raise FileNotFoundError(
            f"{TEST_DATA_PATH} not found. Run `python src/preprocess.py` first."
        )

    df = pd.read_csv(TEST_DATA_PATH)
    X_test = df[FEATURE_COLUMNS]
    y_test = df[TARGET_COLUMN]

    logger.info("Loaded test data from %s with shape %s", TEST_DATA_PATH, df.shape)

    return X_test, y_test


def load_trained_model():
    """
    Load the trained model artifact produced by train.py.
    """
    if not MODEL_PATH.exists():
        logger.error("Trained model not found at %s", MODEL_PATH)
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run `python src/train.py` first."
        )

    return load_object(MODEL_PATH)


# ============================================================
# Evaluate
# ============================================================

def compute_metrics(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Compute the full evaluation metric suite on the test set.
    """
    y_pred = model.predict(X_test)

    # Guard predict_proba the same way train.py does, in case the
    # loaded model doesn't expose class probabilities.
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_scores)
    else:
        roc_auc = float("nan")

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
    }

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    logger.info(
        "Confusion matrix — TN: %d, FP: %d, FN: %d, TP: %d", tn, fp, fn, tp
    )

    logger.info("Test set metrics: %s", metrics)
    print_metrics(metrics)

    return metrics


# ============================================================
# CI/CD Gate
# ============================================================

def check_threshold(metrics: dict) -> bool:
    """
    Gate deployment on EVAL_METRIC vs MIN_RECALL_THRESHOLD.

    EVAL_METRIC is configurable (defaults to "recall") because accuracy
    is meaningless on a ~0.17%-positive dataset — a model that never
    predicts fraud still scores ~99.8% accuracy. The threshold itself
    is named MIN_RECALL_THRESHOLD in config.py but is applied generically
    against whichever metric EVAL_METRIC points to.

    Returns
    -------
    bool
        True if the model passes the gate, False otherwise.
    """
    if EVAL_METRIC not in metrics:
        raise KeyError(
            f"EVAL_METRIC '{EVAL_METRIC}' is not a computed metric. "
            f"Available metrics: {list(metrics.keys())}"
        )

    score = metrics[EVAL_METRIC]
    passed = score >= MIN_RECALL_THRESHOLD

    if passed:
        logger.info(
            "PASSED gate: %s = %.4f >= threshold %.4f",
            EVAL_METRIC, score, MIN_RECALL_THRESHOLD,
        )
    else:
        logger.warning(
            "FAILED gate: %s = %.4f < threshold %.4f",
            EVAL_METRIC, score, MIN_RECALL_THRESHOLD,
        )

    return passed


# ============================================================
# Persist Results
# ============================================================

def save_metrics(metrics: dict) -> None:
    """
    Persist metrics to disk as JSON so CI/CD can inspect or archive them
    as a build artifact, independent of MLflow.
    """
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved evaluation metrics to %s", METRICS_PATH)


# ============================================================
# MLflow
# ============================================================

def log_to_mlflow(metrics: dict, passed: bool) -> None:
    """
    Log the evaluation run to MLflow. Failures to reach the tracking
    server are logged as warnings rather than raised, so CI/CD gating
    isn't blocked by an unreachable MLflow server.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name="evaluation"):
            mlflow.set_tag("stage", "evaluation")
            mlflow.set_tag("gate_passed", str(passed))
            mlflow.log_metrics(metrics)

        logger.info("Evaluation run logged to MLflow at %s", MLFLOW_TRACKING_URI)
    except Exception as exc:
        logger.warning("Could not log evaluation run to MLflow (%s). Continuing without it.", exc)


# ============================================================
# Orchestration
# ============================================================

def run_evaluation() -> bool:
    """
    Run the full evaluation pipeline end to end.

    Returns
    -------
    bool
        True if the model passed the CI/CD gate, False otherwise.
    """
    X_test, y_test = load_test_data()
    model = load_trained_model()

    metrics = compute_metrics(model, X_test, y_test)

    save_metrics(metrics)

    passed = check_threshold(metrics)

    log_to_mlflow(metrics, passed)

    logger.info("Evaluation pipeline completed successfully.")

    return passed


if __name__ == "__main__":
    try:
        gate_passed = run_evaluation()
        if not gate_passed:
            logger.error(
                "Model failed the CI/CD quality gate (%s < %.4f). Exiting with failure.",
                EVAL_METRIC, MIN_RECALL_THRESHOLD,
            )
            sys.exit(1)
        sys.exit(0)
    except Exception as exc:
        logger.exception("Evaluation pipeline failed: %s", exc)
        sys.exit(1)