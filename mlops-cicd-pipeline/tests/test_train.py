"""
test_train.py

Unit tests for src.train.
"""

import math
from pathlib import Path
import numpy as np

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import train
from src.config import (
    DEFAULT_MODEL,
    FEATURE_COLUMNS,
    LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS,
    SUPPORTED_MODELS,
    TARGET_COLUMN,
)


# ============================================================
# Fixtures
#
# NOTE: sample_dataframe is now duplicated across test_preprocess.py
# and test_train.py. Move it into tests/conftest.py so both files
# (and test_model.py / test_api.py, if they need it) share one
# definition instead of drifting out of sync.
# ============================================================

@pytest.fixture
def sample_dataframe():
    """
    Small balanced dataset suitable for model training.
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
# Config Sanity
# ============================================================

def test_default_model_is_supported():
    """
    Config integrity check: DEFAULT_MODEL must actually be one of the
    models train.build_model knows how to construct. Cheap to check,
    and catches a config.py typo before it surfaces as a confusing
    error deep inside build_model.
    """
    assert DEFAULT_MODEL in SUPPORTED_MODELS


# ============================================================
# Load
# ============================================================

def test_load_split_file_not_found_explicit_path():
    """Missing file at an explicitly given path should raise."""

    with pytest.raises(FileNotFoundError):
        train.load_split(Path("does_not_exist.csv"))


def test_load_split_file_not_found_default_path(monkeypatch):
    """
    Same check, but exercising the module-level default path rather
    than an explicit argument, in case load_split has a no-argument
    call form that falls back to TRAIN_DATA_PATH.
    """

    monkeypatch.setattr(train, "TRAIN_DATA_PATH", Path("also_does_not_exist.csv"))

    with pytest.raises(FileNotFoundError):
        train.load_split(train.TRAIN_DATA_PATH)


def test_load_split_success(tmp_path, sample_dataframe):
    X, y = sample_dataframe

    df = X.copy()
    df[TARGET_COLUMN] = y

    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)

    X_loaded, y_loaded = train.load_split(csv_path)

    pd.testing.assert_frame_equal(X_loaded, X)
    pd.testing.assert_series_equal(y_loaded, y, check_names=False)


# ============================================================
# Model
# ============================================================

def test_build_default_model():
    model, params = train.build_model(DEFAULT_MODEL)

    if DEFAULT_MODEL == "RandomForest":
        assert isinstance(model, RandomForestClassifier)
        assert params == RANDOM_FOREST_PARAMS
    else:
        assert isinstance(model, LogisticRegression)
        assert params == LOGISTIC_REGRESSION_PARAMS


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_build_supported_models(model_name):
    model, params = train.build_model(model_name)

    assert model is not None
    assert isinstance(params, dict)


@pytest.mark.parametrize(
    "model_name,expected_params",
    [
        ("RandomForest", RANDOM_FOREST_PARAMS),
        ("LogisticRegression", LOGISTIC_REGRESSION_PARAMS),
    ],
)
def test_build_model_applies_params_to_instance(model_name, expected_params):
    """
    Confirms the returned params dict actually landed on the model
    object itself, not just that build_model *reports* matching
    params. Specifically checks class_weight and random_state, since
    those are the two settings that matter most for this imbalanced
    dataset — a bug here would silently produce an unweighted model
    while the returned params dict still looked correct.
    """

    model, _ = train.build_model(model_name)
    model_params = model.get_params()

    for key in ("class_weight", "random_state"):
        if key in expected_params:
            assert model_params[key] == expected_params[key]


def test_build_model_invalid():
    with pytest.raises(ValueError):
        train.build_model("InvalidModel")


# ============================================================
# Training
# ============================================================

def test_train_model_actually_fits(sample_dataframe):
    """
    hasattr(trained, "predict") is true for every scikit-learn
    estimator whether or not it's fitted, so it can't tell us
    train_model actually called .fit(). Calling .predict() is the
    real check: it raises NotFittedError on an unfitted model.
    """

    X, y = sample_dataframe

    model = RandomForestClassifier(n_estimators=5, random_state=42)

    trained = train.train_model(model, X, y)

    predictions = trained.predict(X)  # raises if not fitted

    assert len(predictions) == len(X)


# ============================================================
# Evaluation
# ============================================================

def test_evaluate_model_keys(sample_dataframe):
    X, y = sample_dataframe

    model = RandomForestClassifier(
        n_estimators=5,
        random_state=42,
    )

    model.fit(X, y)

    metrics = train.evaluate_model(model, X, y)

    expected = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }

    assert expected == set(metrics.keys())


def test_evaluate_model_correctness_on_known_predictions():
    """
    Checks actual metric *values*, not just that the keys exist.
    A DummyModel with hardcoded, known-wrong predictions lets us
    verify the metrics formula itself rather than trusting whatever
    scikit-learn happens to compute.

    y_true:      [0, 0, 1, 1]
    y_pred:      [0, 1, 1, 1]   -> 1 false positive, 0 false negatives

    accuracy  = 3/4 = 0.75
    precision = TP / (TP + FP) = 2 / 3
    recall    = TP / (TP + FN) = 2 / 2 = 1.0
    """

    class DummyModel:
        def predict(self, X):
            return [0, 1, 1, 1]

        def predict_proba(self, X):
    # Probabilities consistent with the hard predictions above
            return np.array([
                [0.9, 0.1],
                [0.4, 0.6],
                [0.2, 0.8],
                [0.1, 0.9],
            ])

    X = pd.DataFrame({"feature": [0, 0, 0, 0]})
    y = pd.Series([0, 0, 1, 1])

    metrics = train.evaluate_model(DummyModel(), X, y)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)


def test_evaluate_model_without_predict_proba(sample_dataframe):
    class DummyModel:
        def predict(self, X):
            return [0] * len(X)

    X, y = sample_dataframe

    metrics = train.evaluate_model(DummyModel(), X, y)

    assert "roc_auc" in metrics
    assert math.isnan(metrics["roc_auc"])


# ============================================================
# MLflow
# ============================================================

def test_log_to_mlflow(monkeypatch):
    calls = []
    logged_params = {}
    logged_metrics = {}

    monkeypatch.setattr(train.mlflow, "set_tracking_uri", lambda uri: calls.append("uri"))
    monkeypatch.setattr(train.mlflow, "set_experiment", lambda exp: calls.append("exp"))
    monkeypatch.setattr(train.mlflow, "set_tag", lambda *args, **kwargs: calls.append("tag"))
    monkeypatch.setattr(train.mlflow, "log_param", lambda *args, **kwargs: calls.append("param"))

    def fake_log_params(params):
        calls.append("params")
        logged_params.update(params)

    def fake_log_metrics(metrics):
        calls.append("metrics")
        logged_metrics.update(metrics)

    monkeypatch.setattr(train.mlflow, "log_params", fake_log_params)
    monkeypatch.setattr(train.mlflow, "log_metrics", fake_log_metrics)
    monkeypatch.setattr(train.mlflow.sklearn, "log_model", lambda *args, **kwargs: calls.append("model"))

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(train.mlflow, "start_run", lambda: DummyRun())

    model = RandomForestClassifier()
    input_params = {"n_estimators": 10}
    input_metrics = {"accuracy": 1.0}

    train.log_to_mlflow(
        model,
        "RandomForest",
        input_params,
        input_metrics,
    )

    assert "uri" in calls
    assert "metrics" in calls
    assert "model" in calls
    # Confirm the exact params/metrics passed in were the ones logged,
    # not just that log_params/log_metrics were called with *something*.
    assert logged_params == input_params
    assert logged_metrics == input_metrics


def test_log_to_mlflow_failure_does_not_raise(monkeypatch):
    """
    An MLflow outage (e.g. tracking server unreachable) must not crash
    the training run — model training succeeded and the model should
    still be saved locally even if experiment tracking failed.
    """

    monkeypatch.setattr(
        train.mlflow,
        "set_tracking_uri",
        lambda uri: (_ for _ in ()).throw(RuntimeError("MLflow Down")),
    )

    model = RandomForestClassifier()

    # should not raise
    train.log_to_mlflow(
        model,
        "RandomForest",
        {},
        {},
    )


# ============================================================
# Pipeline
# ============================================================

def test_run_training(monkeypatch):
    calls = []

    X = pd.DataFrame()
    y = pd.Series(dtype=int)

    monkeypatch.setattr(
        train,
        "load_split",
        lambda path: (calls.append("load"), (X, y))[1],
    )

    monkeypatch.setattr(
        train,
        "build_model",
        lambda name: (calls.append("build"), (object(), {}))[1],
    )

    monkeypatch.setattr(
        train,
        "train_model",
        lambda model, X, y: (calls.append("train"), model)[1],
    )

    monkeypatch.setattr(
        train,
        "evaluate_model",
        lambda model, X, y: (calls.append("evaluate"), {})[1],
    )

    monkeypatch.setattr(
        train,
        "log_to_mlflow",
        lambda *args, **kwargs: calls.append("mlflow"),
    )

    monkeypatch.setattr(
        train,
        "save_object",
        lambda *args, **kwargs: calls.append("save"),
    )

    train.run_training()

    assert calls == [
        "load",
        "load",
        "build",
        "train",
        "evaluate",
        "mlflow",
        "save",
    ]