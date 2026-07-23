"""
main.py

FastAPI application for the Credit Card Fraud Detection service.

Endpoints:
- GET  /health    -> service + model status
- POST /predict   -> fraud prediction for a single transaction
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.predictor import predictor
from api.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.config import HOST, PORT, DEBUG
from src.utils import logger

APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model once at startup, so each request doesn't pay the
    joblib-load cost. If loading fails, the app still starts (so
    /health can report the failure) but /predict will 503 until fixed.
    """
    try:
        predictor.load()
    except Exception as exc:
        logger.exception("Failed to load model at startup: %s", exc)

    yield

    logger.info("Shutting down fraud detection API.")


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Serves fraud predictions from the trained MLOps pipeline model.",
    version=APP_VERSION,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Report service health and whether the model is loaded.
    """
    return HealthResponse(
        status="ok" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        version=APP_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict whether a transaction is fraudulent.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check /health and server logs.",
        )

    try:
        result = predictor.predict(request)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    return PredictionResponse(**result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=HOST, port=PORT, reload=DEBUG)