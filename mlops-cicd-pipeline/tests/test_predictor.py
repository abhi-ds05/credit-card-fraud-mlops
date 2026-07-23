"""
test_predictor.py

Unit tests for api.predictor.

NOTE: several tests below assume api.predictor exposes MODEL_PATH and
SCALER_PATH (imported from src.config, matching the pattern already
used in src/evaluate.py). If predictor.py's real attribute names
differ, update the monkeypatch targets accordingly — this is safer
than the previous approach of patching Path.exists globally, but it
does depend on those names existing.
"""

import pandas as pd
import pytest

from api.predictor import FraudPredictor
from src.config import FEATURE_COLUMNS


# ============================================================
# Dummy Objects
#
# NOTE: these are duplicated in spirit with dummy models used in
# test_train.py / test_evaluate.py. Consider a shared conftest.py
# (or a tests/fixtures.py module) if api tests and src tests end up
# needing the same dummy model/scaler shapes.
# ============================================================

class DummyRequest:
    """
    Mimics PredictionRequest.
    """

    def model_dump(self):
        data = {
            "Time": 1.0,
            "Amount": 100.0,
        }

        for i in range(1, 29):
            data[f"V{i}"] = float(i)

        return data


class DummyScaler:
    def transform(self, X):
        return X


class DummyScalerModifies:
    """
    Unlike DummyScaler, this one actually changes the input — needed
    to prove preprocess() really calls scaler.transform() rather than
    silently skipping it. A no-op scaler can't distinguish "scaling
    was applied" from "scaling was never called".
    """

    def transform(self, X):
        X = X.copy()
        X["Amount"] = X["Amount"] * 0
        return X


class DummyModel:
    def predict(self, X):
        return [1]

    def predict_proba(self, X):
        return [[0.05, 0.95]]


class DummyModelNoProb:
    def predict(self, X):
        return [0]


class DummyModelNoProbFraud:
    """
    Covers the 'no predict_proba' branch for the *fraud* (1) case —
    the original suite only tested this branch for the legitimate (0)
    case, leaving the positive-class path untested.
    """

    def predict(self, X):
        return [1]


# ============================================================
# Constructor
# ============================================================

def test_predictor_initial_state():
    predictor = FraudPredictor()

    assert predictor.model is None
    assert predictor.scaler is None
    assert predictor.is_loaded is False


# ============================================================
# Load
# ============================================================

def test_load_missing_model(monkeypatch, tmp_path):
    """
    Uses real (nonexistent) paths in tmp_path instead of globally
    monkeypatching Path.exists — a missing file naturally reports
    exists() == False without needing to patch the class at all.
    """

    predictor = FraudPredictor()

    monkeypatch.setattr("api.predictor.MODEL_PATH", tmp_path / "missing_model.joblib")
    monkeypatch.setattr("api.predictor.SCALER_PATH", tmp_path / "missing_scaler.joblib")

    with pytest.raises(FileNotFoundError):
        predictor.load()


def test_load_model_and_scaler(monkeypatch, tmp_path):
    predictor = FraudPredictor()

    model_path = tmp_path / "model.joblib"
    scaler_path = tmp_path / "scaler.joblib"
    model_path.write_bytes(b"placeholder")
    scaler_path.write_bytes(b"placeholder")

    monkeypatch.setattr("api.predictor.MODEL_PATH", model_path)
    monkeypatch.setattr("api.predictor.SCALER_PATH", scaler_path)

    def fake_load(path):
        # Match by which path was actually requested, not call order —
        # robust to load() checking/loading model and scaler in either
        # sequence.
        if path == model_path:
            return DummyModel()
        if path == scaler_path:
            return DummyScaler()
        raise ValueError(f"Unexpected path passed to joblib.load: {path}")

    monkeypatch.setattr("api.predictor.joblib.load", fake_load)

    predictor.load()

    assert predictor.is_loaded
    assert isinstance(predictor.model, DummyModel)
    assert isinstance(predictor.scaler, DummyScaler)


def test_load_without_scaler(monkeypatch, tmp_path):
    """
    Model exists, scaler file does not — load() should still succeed
    with scaler left as None (matching the design where scaling is
    optional at inference time).
    """

    predictor = FraudPredictor()

    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"placeholder")
    scaler_path = tmp_path / "missing_scaler.joblib"  # never created

    monkeypatch.setattr("api.predictor.MODEL_PATH", model_path)
    monkeypatch.setattr("api.predictor.SCALER_PATH", scaler_path)
    monkeypatch.setattr("api.predictor.joblib.load", lambda path: DummyModel())

    predictor.load()

    assert predictor.is_loaded
    assert predictor.scaler is None


# ============================================================
# Preprocess
# ============================================================

def test_preprocess_without_scaler():
    predictor = FraudPredictor()

    request = DummyRequest()

    df = predictor.preprocess(request)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1


def test_preprocess_column_order_matches_training_features():
    """
    Critical correctness check: scikit-learn models match input
    features by column *position*, not name. If preprocess() ever
    produces columns in a different order than FEATURE_COLUMNS (the
    order the model was trained on), predictions become silently
    wrong instead of raising an error — there's no exception to catch
    this, only a test.
    """

    predictor = FraudPredictor()

    request = DummyRequest()

    df = predictor.preprocess(request)

    assert list(df.columns) == list(FEATURE_COLUMNS)


def test_preprocess_with_scaler_actually_applies_transform():
    """
    DummyScaler (a no-op) can't prove transform() was really called.
    DummyScalerModifies changes Amount to 0, so if the returned
    DataFrame doesn't reflect that, either the scaler wasn't invoked
    or its output was discarded.
    """

    predictor = FraudPredictor()
    predictor.scaler = DummyScalerModifies()

    request = DummyRequest()  # Amount = 100.0

    df = predictor.preprocess(request)

    assert df["Amount"].iloc[0] == 0


# ============================================================
# Predict
# ============================================================

def test_predict_requires_loaded_model():
    predictor = FraudPredictor()

    with pytest.raises(RuntimeError):
        predictor.predict(DummyRequest())


def test_predict_success(monkeypatch):
    predictor = FraudPredictor()

    predictor.model = DummyModel()
    predictor.scaler = None
    predictor._loaded = True

    result = predictor.predict(DummyRequest())

    assert result["prediction"] == 1
    assert result["probability"] == pytest.approx(0.95)
    assert result["label"] == "Fraudulent Transaction"


def test_predict_without_probability_legitimate():
    predictor = FraudPredictor()

    predictor.model = DummyModelNoProb()
    predictor.scaler = None
    predictor._loaded = True

    result = predictor.predict(DummyRequest())

    assert result["prediction"] == 0
    assert result["probability"] == 0.0
    assert result["label"] == "Legitimate Transaction"


def test_predict_without_probability_fraud():
    """
    Covers the fraud (1) branch of the no-predict_proba path, which
    the original suite never exercised — only the legitimate (0) case
    was tested for a model lacking predict_proba.
    """

    predictor = FraudPredictor()

    predictor.model = DummyModelNoProbFraud()
    predictor.scaler = None
    predictor._loaded = True

    result = predictor.predict(DummyRequest())

    assert result["prediction"] == 1
    assert result["label"] == "Fraudulent Transaction"


# ============================================================
# Property
# ============================================================

def test_is_loaded_requires_model():
    predictor = FraudPredictor()

    predictor._loaded = True
    predictor.model = None

    assert predictor.is_loaded is False

    predictor.model = DummyModel()

    assert predictor.is_loaded is True


# ============================================================
# Integration
# ============================================================

def test_predict_calls_preprocess(monkeypatch):
    predictor = FraudPredictor()

    predictor.model = DummyModel()
    predictor._loaded = True

    called = {"preprocess": False}

    def fake_preprocess(request):
        called["preprocess"] = True
        return pd.DataFrame([[0] * 30])

    monkeypatch.setattr(predictor, "preprocess", fake_preprocess)

    predictor.predict(DummyRequest())

    assert called["preprocess"]