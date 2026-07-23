"""
predictor.py

Model loading and inference logic for the Credit Card Fraud Detection API.

Replaces the earlier model_loader.py with the same responsibilities:
- Load the trained model + fitted scaler once
- Preprocess incoming requests identically to preprocess.py
- Run inference and return a prediction dict
"""

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from api.schemas import PredictionRequest
from src.config import FEATURE_COLUMNS, MODEL_PATH, SCALER_PATH, SCALE_AMOUNT, SCALE_TIME
from src.utils import logger


# ============================================================
# Predictor Class
# ============================================================

class FraudPredictor:
    """
    Handles loading the trained model and performing inference.

    Loading is NOT done in __init__ (unlike the original predictor.py)
    so that importing this module never crashes if the model/scaler
    don't exist yet. main.py calls load() explicitly during FastAPI's
    startup lifespan, letting the app boot in a "degraded" state and
    report that clearly via /health instead of failing to import.
    """

    def __init__(self) -> None:
        self.model: Optional[object] = None
        self.scaler: Optional[object] = None
        self._loaded = False

    # --------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None

    # --------------------------------------------------------
    def load(self) -> None:
        """
        Load the trained model and, if present, the fitted scaler.

        MODEL_PATH comes from config.py (models/credit_card_fraud_model.joblib
        by default) so this always matches whatever train.py just saved,
        regardless of which SUPPORTED_MODELS option was trained.
        """
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}. Run `python src/train.py` first."
            )

        self.model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully from %s", MODEL_PATH)

        if Path(SCALER_PATH).exists():
            self.scaler = joblib.load(SCALER_PATH)
            logger.info("Scaler loaded successfully from %s", SCALER_PATH)
        else:
            self.scaler = None
            logger.warning(
                "Scaler not found at %s — proceeding without scaling. "
                "Only correct if the model was trained on unscaled features.",
                SCALER_PATH,
            )

        self._loaded = True

    # --------------------------------------------------------
    def preprocess(self, request: PredictionRequest) -> pd.DataFrame:
        """
        Convert a request into a model-ready DataFrame, applying the
        exact same scaling preprocess.py applied at training time.

        Only the columns actually scaled during training (per
        config.SCALE_AMOUNT / config.SCALE_TIME) are transformed here.
        Scaling both Time and Amount unconditionally would break at
        runtime whenever SCALE_TIME is False, since the scaler was
        fitted on fewer columns than that would try to transform.
        """
        data = request.model_dump()
        df = pd.DataFrame([data])
        df = df[FEATURE_COLUMNS]

        if self.scaler is not None:
            columns_to_scale = []
            if SCALE_AMOUNT:
                columns_to_scale.append("Amount")
            if SCALE_TIME:
                columns_to_scale.append("Time")

            if columns_to_scale:
                df[columns_to_scale] = self.scaler.transform(df[columns_to_scale])

        return df

    # --------------------------------------------------------
    def predict(self, request: PredictionRequest) -> dict:
        """
        Perform fraud prediction for a single transaction.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() before predict().")

        df = self.preprocess(request)

        prediction = int(self.model.predict(df)[0])

        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(df)[0][1])
        else:
            # Fall back to the hard prediction if the model can't
            # produce a probability, rather than failing the request.
            probability = float(prediction)

        label = "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"

        logger.info(
            "Prediction completed. Prediction=%d, Probability=%.4f",
            prediction, probability,
        )

        return {
            "prediction": prediction,
            "probability": round(probability, 6),
            "label": label,
        }


# ============================================================
# Module-level singleton
# ============================================================

# Lazily loaded: created here, but .load() is called explicitly from
# main.py's startup lifespan, not at import time.
predictor = FraudPredictor()