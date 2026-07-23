"""
test_api.py

Unit tests for api.main.
"""

from fastapi.testclient import TestClient
import pytest

from api.main import app
from api.predictor import predictor


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def client():
    """
    Shared TestClient for endpoint tests that don't need to trigger
    lifespan startup/shutdown explicitly. Lifespan-specific tests
    still use `with TestClient(app):` directly, since that context
    manager is what actually fires the startup/shutdown events.
    """
    return TestClient(app)


def valid_payload():
    """
    Valid PredictionRequest payload.
    """
    payload = {
        "Time": 1.0,
        "Amount": 100.0,
    }

    for i in range(1, 29):
        payload[f"V{i}"] = float(i)

    return payload


# ============================================================
# Health Endpoint
# ============================================================

def test_health_loaded(monkeypatch, client):
    monkeypatch.setattr(predictor, "_loaded", True)
    monkeypatch.setattr(predictor, "model", object())

    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["version"] == app.version


def test_health_degraded(monkeypatch, client):
    monkeypatch.setattr(predictor, "_loaded", False)
    monkeypatch.setattr(predictor, "model", None)

    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert body["version"] == app.version


# ============================================================
# Predict Endpoint
# ============================================================

def test_predict_success(monkeypatch, client):
    monkeypatch.setattr(predictor, "_loaded", True)
    monkeypatch.setattr(predictor, "model", object())

    monkeypatch.setattr(
        predictor,
        "predict",
        lambda request: {
            "prediction": 1,
            "probability": 0.95,
            "label": "Fraudulent Transaction",
        },
    )

    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200

    body = response.json()

    assert body["prediction"] == 1
    assert body["probability"] == pytest.approx(0.95)
    assert body["label"] == "Fraudulent Transaction"
    # Contract check: no extra or missing fields in the response shape.
    assert set(body.keys()) == {"prediction", "probability", "label"}


def test_predict_forwards_parsed_request_data(monkeypatch, client):
    """
    The other predict tests use a lambda that ignores its `request`
    argument entirely, so nothing actually confirms the endpoint
    forwards the parsed PredictionRequest through to predictor.predict
    correctly. This captures what's actually passed and checks it
    reflects the submitted payload, not just that *some* call happened.
    """

    monkeypatch.setattr(predictor, "_loaded", True)
    monkeypatch.setattr(predictor, "model", object())

    captured = {}

    def fake_predict(request):
        captured["request"] = request
        return {"prediction": 0, "probability": 0.1, "label": "Legitimate Transaction"}

    monkeypatch.setattr(predictor, "predict", fake_predict)

    payload = valid_payload()
    payload["Amount"] = 12345.0

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "request" in captured
    assert captured["request"].Amount == 12345.0


def test_predict_model_not_loaded(monkeypatch, client):
    monkeypatch.setattr(predictor, "_loaded", False)
    monkeypatch.setattr(predictor, "model", None)

    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 503
    assert "Model is not loaded" in response.json()["detail"]


def test_predict_internal_error(monkeypatch, client):
    monkeypatch.setattr(predictor, "_loaded", True)
    monkeypatch.setattr(predictor, "model", object())

    def fail(request):
        raise RuntimeError("boom: internal file path /etc/secrets")

    monkeypatch.setattr(predictor, "predict", fail)

    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 500

    body_text = response.text

    assert response.json()["detail"] == "Prediction failed."
    # Security check: the raw exception message must never reach the
    # client — an endpoint that "handles" the error but still echoes
    # internals back is effectively an information-disclosure bug.
    assert "boom" not in body_text
    assert "/etc/secrets" not in body_text


# ============================================================
# Validation
# ============================================================

def test_predict_invalid_request_empty_body(client):
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_predict_missing_single_field(client):
    payload = valid_payload()
    payload.pop("Amount")

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_invalid_field_type(client):
    """
    Distinct from the missing-field test above: this checks Pydantic
    actually rejects the *wrong type* for a field, not just a missing
    one. A schema that only enforces presence but not type would pass
    every previous validation test while still accepting garbage data.
    """

    payload = valid_payload()
    payload["Amount"] = "not-a-number"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


# ============================================================
# Lifespan
# ============================================================

def test_startup_load_called(monkeypatch):
    calls = []

    monkeypatch.setattr(predictor, "load", lambda: calls.append("load"))

    with TestClient(app):
        pass

    assert calls == ["load"]


def test_startup_load_failure_does_not_crash(monkeypatch):
    def fail():
        raise RuntimeError("cannot load")

    monkeypatch.setattr(predictor, "load", fail)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200


# ============================================================
# OpenAPI
# ============================================================

def test_openapi_schema(client):
    response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()

    assert "/health" in schema["paths"]
    assert "/predict" in schema["paths"]


def test_docs_available(client):
    response = client.get("/docs")

    assert response.status_code == 200


# ============================================================
# API Metadata
# ============================================================

def test_application_metadata():
    assert app.title == "Credit Card Fraud Detection API"
    assert app.version == "1.0.0"
    assert "fraud" in app.description.lower()