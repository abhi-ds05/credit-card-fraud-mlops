"""
test_evaluate.py

Unit tests for src.evaluate.
"""

import json
import math
from pathlib import Path
import numpy as np

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src import evaluate
from src.config import FEATURE_COLUMNS, TARGET_COLUMN


# ============================================================
# Fixtures
#
# NOTE: sample_dataframe is now duplicated across test_preprocess.py,
# test_train.py, and test_evaluate.py. Move it into tests/conftest.py
# so all three (and test_api.py, if needed) share one definition.
# ============================================================

@pytest.fixture
def sample_dataframe():
    """
    Small balanced dataset suitable for evaluation.
    """
    rows = 100

    data = {
        "Time": list(range(rows)),
        "Amount": [float(i) for i in range(rows)],
    }

    for i in range(1, 29):
        data[f"V{i}"] = [float(i)] * rows

    data[TARGET_COLUMN] = [0] * 50 + [1] * 50

    df = pd.DataFrame(data)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return X, y


# ============================================================
# Load
# ============================================================

def test_load_test_data_file_not_found(monkeypatch):
    monkeypatch.setattr(evaluate, "TEST_DATA_PATH", Path("missing.csv"))

    with pytest.raises(FileNotFoundError):
        evaluate.load_test_data()


def test_load_test_data_success(monkeypatch, tmp_path, sample_dataframe):
    X, y = sample_dataframe

    df = X.copy()
    df[TARGET_COLUMN] = y

    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)

    monkeypatch.setattr(evaluate, "TEST_DATA_PATH", csv_path)

    X_loaded, y_loaded = evaluate.load_test_data()

    pd.testing.assert_frame_equal(X_loaded, X)
    pd.testing.assert_series_equal(y_loaded, y, check_names=False)


def test_load_trained_model_not_found(monkeypatch):
    monkeypatch.setattr(evaluate, "MODEL_PATH", Path("missing.joblib"))

    with pytest.raises(FileNotFoundError):
        evaluate.load_trained_model()


def test_load_trained_model(monkeypatch, tmp_path):
    """
    Uses a real file in tmp_path rather than globally monkeypatching
    Path.exists — patching the builtin for every Path instance risks
    masking bugs in any other path check load_trained_model might do
    internally, since everything would report "exists" regardless.
    """

    model_path = tmp_path / "dummy.joblib"
    model_path.write_bytes(b"not a real model, just needs to exist")

    model = object()

    monkeypatch.setattr(evaluate, "MODEL_PATH", model_path)
    monkeypatch.setattr(evaluate, "load_object", lambda path: model)

    assert evaluate.load_trained_model() is model


# ============================================================
# Metrics
# ============================================================

def test_compute_metrics_keys(sample_dataframe):
    X, y = sample_dataframe

    model = RandomForestClassifier(
        n_estimators=5,
        random_state=42,
    )

    model.fit(X, y)

    metrics = evaluate.compute_metrics(model, X, y)

    expected = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }

    assert expected == set(metrics.keys())


def test_compute_metrics_correctness_on_known_predictions():
    """
    Checks actual metric *values* against a hand-calculated result,
    not just that the expected keys exist. Mirrors the equivalent
    check in test_train.py for evaluate_model — the two shouldn't be
    allowed to silently drift into computing metrics differently.

    y_true: [0, 0, 1, 1]
    y_pred: [0, 1, 1, 1]  -> 1 false positive, 0 false negatives

    accuracy  = 3/4 = 0.75
    precision = TP / (TP + FP) = 2 / 3
    recall    = TP / (TP + FN) = 2 / 2 = 1.0
    """

    class DummyModel:
        def predict(self, X):
            return [0, 1, 1, 1]

        def predict_proba(self, X):
            return np.array([
                [0.9, 0.1],
                [0.4, 0.6],
                [0.2, 0.8],
                [0.1, 0.9],
            ])

    X = pd.DataFrame({"feature": [0, 0, 0, 0]})
    y = pd.Series([0, 0, 1, 1])

    metrics = evaluate.compute_metrics(DummyModel(), X, y)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)


def test_compute_metrics_without_predict_proba(sample_dataframe):
    class DummyModel:
        def predict(self, X):
            return [0] * len(X)

    X, y = sample_dataframe

    metrics = evaluate.compute_metrics(DummyModel(), X, y)

    assert math.isnan(metrics["roc_auc"])


# ============================================================
# Threshold
#
# NOTE: MIN_RECALL_THRESHOLD is named recall-specifically in
# config.py, but EVAL_METRIC is meant to be configurable (config.py's
# comment mentions "recall or f1"). If check_threshold is meant to
# gate on whatever EVAL_METRIC is set to, the threshold constant
# should probably be renamed to something generic like
# MIN_METRIC_THRESHOLD to avoid confusion — worth confirming this
# against the real evaluate.py implementation.
# ============================================================

def test_check_threshold_pass(monkeypatch):
    monkeypatch.setattr(evaluate, "EVAL_METRIC", "recall")
    monkeypatch.setattr(evaluate, "MIN_RECALL_THRESHOLD", 0.80)

    assert evaluate.check_threshold({"recall": 0.90})


def test_check_threshold_fail(monkeypatch):
    monkeypatch.setattr(evaluate, "EVAL_METRIC", "recall")
    monkeypatch.setattr(evaluate, "MIN_RECALL_THRESHOLD", 0.80)

    assert not evaluate.check_threshold({"recall": 0.60})


def test_check_threshold_exactly_at_boundary(monkeypatch):
    """
    A metric exactly equal to the threshold should pass (>=, not >).
    Off-by-one-comparison-operator bugs are easy to introduce and
    easy to miss without a boundary test.
    """
    monkeypatch.setattr(evaluate, "EVAL_METRIC", "recall")
    monkeypatch.setattr(evaluate, "MIN_RECALL_THRESHOLD", 0.80)

    assert evaluate.check_threshold({"recall": 0.80})


def test_check_threshold_invalid_metric(monkeypatch):
    monkeypatch.setattr(evaluate, "EVAL_METRIC", "invalid")

    with pytest.raises(KeyError):
        evaluate.check_threshold({"accuracy": 1.0})


# ============================================================
# Save Metrics
# ============================================================

def test_save_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(
        evaluate,
        "METRICS_PATH",
        tmp_path / "evaluation_metrics.json",
    )

    metrics = {
        "accuracy": 1.0,
        "recall": 0.95,
    }

    evaluate.save_metrics(metrics)

    assert (tmp_path / "evaluation_metrics.json").exists()

    with open(tmp_path / "evaluation_metrics.json") as f:
        loaded = json.load(f)

    assert loaded == metrics


def test_save_metrics_creates_parent_directories(monkeypatch, tmp_path):
    """
    Mirrors the equivalent check for save_object in utils.py — if
    METRICS_PATH points somewhere that doesn't exist yet (e.g. a
    fresh clone of the repo with no models/ folder), save_metrics
    should create it rather than raising.
    """

    nested_path = tmp_path / "nested" / "sub" / "evaluation_metrics.json"

    monkeypatch.setattr(evaluate, "METRICS_PATH", nested_path)

    evaluate.save_metrics({"accuracy": 1.0})

    assert nested_path.exists()


# ============================================================
# MLflow
# ============================================================

def test_log_to_mlflow_when_passed(monkeypatch):
    calls = []
    logged_metrics = {}

    monkeypatch.setattr(
        evaluate.mlflow,
        "set_tracking_uri",
        lambda uri: calls.append("uri"),
    )

    monkeypatch.setattr(
        evaluate.mlflow,
        "set_experiment",
        lambda exp: calls.append("experiment"),
    )

    logged_tags = {}

    def fake_set_tag(key, value):
        calls.append("tag")
        logged_tags[key] = value

    monkeypatch.setattr(evaluate.mlflow, "set_tag", fake_set_tag)

    def fake_log_metrics(metrics):
        calls.append("metrics")
        logged_metrics.update(metrics)

    monkeypatch.setattr(evaluate.mlflow, "log_metrics", fake_log_metrics)

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(evaluate.mlflow, "start_run", lambda **kwargs: DummyRun())

    metrics = {"accuracy": 1.0}

    evaluate.log_to_mlflow(metrics, True)

    assert "uri" in calls
    assert "metrics" in calls
    assert logged_metrics == metrics
    # At least one tag should reflect the passing gate result.
    assert any(str(v).lower() in ("true", "passed", "pass") for v in logged_tags.values())


def test_log_to_mlflow_when_failed(monkeypatch):
    """
    The passed=False branch was previously untested entirely — this
    confirms metrics still get logged even when the quality gate
    fails, since a failed run is exactly when you most need the
    numbers recorded for debugging.
    """

    calls = []
    logged_metrics = {}
    logged_tags = {}

    monkeypatch.setattr(evaluate.mlflow, "set_tracking_uri", lambda uri: calls.append("uri"))
    monkeypatch.setattr(evaluate.mlflow, "set_experiment", lambda exp: calls.append("experiment"))

    def fake_set_tag(key, value):
        calls.append("tag")
        logged_tags[key] = value

    monkeypatch.setattr(evaluate.mlflow, "set_tag", fake_set_tag)

    def fake_log_metrics(metrics):
        calls.append("metrics")
        logged_metrics.update(metrics)

    monkeypatch.setattr(evaluate.mlflow, "log_metrics", fake_log_metrics)

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(evaluate.mlflow, "start_run", lambda **kwargs: DummyRun())

    metrics = {"accuracy": 0.50}

    evaluate.log_to_mlflow(metrics, False)

    assert "metrics" in calls
    assert logged_metrics == metrics
    assert any(str(v).lower() in ("false", "failed", "fail") for v in logged_tags.values())


def test_log_to_mlflow_failure_does_not_raise(monkeypatch):
    """
    An MLflow outage must not crash evaluation — the metrics were
    already computed and saved locally before this is ever called.
    """

    monkeypatch.setattr(
        evaluate.mlflow,
        "set_tracking_uri",
        lambda uri: (_ for _ in ()).throw(RuntimeError("MLflow Down")),
    )

    evaluate.log_to_mlflow({}, True)  # should not raise


# ============================================================
# Pipeline
# ============================================================

def test_run_evaluation_gate_passes(monkeypatch):
    calls = []

    X = pd.DataFrame()
    y = pd.Series(dtype=int)

    monkeypatch.setattr(
        evaluate,
        "load_test_data",
        lambda: (calls.append("load_data"), (X, y))[1],
    )

    monkeypatch.setattr(
        evaluate,
        "load_trained_model",
        lambda: (calls.append("load_model"), object())[1],
    )

    monkeypatch.setattr(
        evaluate,
        "compute_metrics",
        lambda model, X, y: (calls.append("metrics"), {"recall": 1.0})[1],
    )

    monkeypatch.setattr(
        evaluate,
        "save_metrics",
        lambda metrics: calls.append("save"),
    )

    monkeypatch.setattr(
        evaluate,
        "check_threshold",
        lambda metrics: (calls.append("gate"), True)[1],
    )

    monkeypatch.setattr(
        evaluate,
        "log_to_mlflow",
        lambda metrics, passed: calls.append("mlflow"),
    )

    assert evaluate.run_evaluation() is True

    assert calls == [
        "load_data",
        "load_model",
        "metrics",
        "save",
        "gate",
        "mlflow",
    ]


def test_run_evaluation_gate_fails(monkeypatch):
    """
    When the model misses the threshold, run_evaluation should still
    save metrics and log to MLflow (so the failing run is visible and
    debuggable), but must return False so CI can fail the pipeline.
    """

    calls = []

    X = pd.DataFrame()
    y = pd.Series(dtype=int)

    monkeypatch.setattr(
        evaluate,
        "load_test_data",
        lambda: (calls.append("load_data"), (X, y))[1],
    )

    monkeypatch.setattr(
        evaluate,
        "load_trained_model",
        lambda: (calls.append("load_model"), object())[1],
    )

    monkeypatch.setattr(
        evaluate,
        "compute_metrics",
        lambda model, X, y: (calls.append("metrics"), {"recall": 0.10})[1],
    )

    monkeypatch.setattr(
        evaluate,
        "save_metrics",
        lambda metrics: calls.append("save"),
    )

    monkeypatch.setattr(
        evaluate,
        "check_threshold",
        lambda metrics: (calls.append("gate"), False)[1],
    )

    monkeypatch.setattr(
        evaluate,
        "log_to_mlflow",
        lambda metrics, passed: calls.append("mlflow"),
    )

    assert evaluate.run_evaluation() is False

    assert "save" in calls
    assert "mlflow" in calls