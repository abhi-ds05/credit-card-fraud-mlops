"""
conftest.py

Shared pytest fixtures for the Credit Card Fraud Detection
MLOps Pipeline.

These fixtures are automatically available to all tests.
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from api.main import app
from api.predictor import predictor
from src.config import FEATURE_COLUMNS, SCALE_AMOUNT, SCALE_TIME


# ============================================================
# Random Seed
# ============================================================

@pytest.fixture(autouse=True)
def set_random_seed():
    """
    Ensure reproducible tests. Runs before every test automatically.
    """
    np.random.seed(42)


# ============================================================
# Sample Transaction
# ============================================================

@pytest.fixture
def sample_transaction():
    """
    Sample transaction payload for prediction tests. Matches
    PredictionRequest exactly (Time, V1-V28, Amount) — the schema
    uses extra="forbid", so any additional/missing keys here would
    fail request validation.
    """
    transaction = {
        "Time": 0.0,
        "Amount": 149.62,
    }
    for i in range(1, 29):
        transaction[f"V{i}"] = float(np.random.normal())
    return transaction


# ============================================================
# Sample Dataset
# ============================================================

@pytest.fixture
def sample_dataframe():
    """
    Small dataframe for preprocessing tests, shaped like the real
    creditcard.csv (Time, V1-V28, Amount, Class).
    """
    np.random.seed(42)
    df = pd.DataFrame({
        "Time": np.random.rand(100),
        "Amount": np.random.rand(100) * 1000,
        "Class": np.random.randint(0, 2, 100),
    })
    for i in range(1, 29):
        df[f"V{i}"] = np.random.randn(100)
    return df


# ============================================================
# Temporary Directory
# ============================================================

@pytest.fixture
def temp_dir(tmp_path):
    """
    Temporary directory for saving models and artifacts.
    """
    return tmp_path


# ============================================================
# Dummy Scaler
# ============================================================

@pytest.fixture
def dummy_scaler(temp_dir):
    """
    Create and save a StandardScaler fitted on exactly the columns
    preprocess.py/predictor.py actually scale (per config.SCALE_AMOUNT /
    config.SCALE_TIME), not a hardcoded pair. Fitting on the wrong
    number of columns causes a shape-mismatch at transform() time,
    which is what happened in the original fixture (always fit on 2
    columns even when SCALE_TIME is False and only 1 column — Amount —
    is scaled in this project's default config).
    """
    columns_to_scale = []
    if SCALE_AMOUNT:
        columns_to_scale.append("Amount")
    if SCALE_TIME:
        columns_to_scale.append("Time")

    # Fall back to a single dummy column if neither flag is set, so the
    # fixture still produces a usable (if unused) scaler in that case.
    n_columns = len(columns_to_scale) or 1

    scaler = StandardScaler()
    X = np.random.rand(100, n_columns)
    scaler.fit(X)

    scaler_path = temp_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    return scaler


# ============================================================
# Dummy Model
# ============================================================

@pytest.fixture
def dummy_model(temp_dir):
    """
    Create and save a trained RandomForest model, using exactly
    len(FEATURE_COLUMNS) features so it lines up with the real
    Time + V1-V28 + Amount feature set predictor.py builds.
    """
    n_features = len(FEATURE_COLUMNS)
    X = np.random.rand(100, n_features)
    y = np.random.randint(0, 2, 100)

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )
    model.fit(X, y)

    model_path = temp_dir / "model.joblib"
    joblib.dump(model, model_path)

    return model


# ============================================================
# Mocked Predictor (loaded state)
# ============================================================

@pytest.fixture
def mock_predictor(monkeypatch, dummy_model, dummy_scaler):
    """
    Force the real `predictor` singleton (imported by api.main) into a
    loaded state backed by dummy_model/dummy_scaler, without touching
    disk or config paths.

    `load` is patched to a no-op FIRST so that FastAPI's startup
    lifespan (which calls predictor.load()) can never overwrite this
    with a real-or-failed load, regardless of whether a real trained
    model happens to exist on the machine running the tests. This
    makes API tests deterministic in any environment (dev, CI, fresh
    checkout with no models/ contents yet).
    """
    monkeypatch.setattr(predictor, "load", lambda: None)
    monkeypatch.setattr(predictor, "model", dummy_model)
    monkeypatch.setattr(predictor, "scaler", dummy_scaler)
    monkeypatch.setattr(predictor, "_loaded", True)

    return predictor


# ============================================================
# Mocked Predictor (unloaded state)
# ============================================================

@pytest.fixture
def unloaded_predictor(monkeypatch):
    """
    Force the predictor singleton into an unloaded state, for testing
    the /health "degraded" response and /predict's 503 path.
    """
    monkeypatch.setattr(predictor, "load", lambda: None)
    monkeypatch.setattr(predictor, "model", None)
    monkeypatch.setattr(predictor, "scaler", None)
    monkeypatch.setattr(predictor, "_loaded", False)

    return predictor


# ============================================================
# FastAPI Test Clients
# ============================================================

@pytest.fixture
def client(mock_predictor):
    """
    FastAPI test client with a loaded (dummy) model. Function-scoped
    (not session-scoped) because it depends on monkeypatch, which is
    itself function-scoped — a session-scoped client would silently
    reuse whatever state the first test happened to set.

    Uses TestClient as a context manager so the app's lifespan (which
    calls predictor.load(), a no-op here thanks to mock_predictor)
    actually runs its startup/shutdown events.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def degraded_client(unloaded_predictor):
    """
    FastAPI test client with no model loaded, for testing degraded
    /health and 503 /predict behavior.
    """
    with TestClient(app) as test_client:
        yield test_client