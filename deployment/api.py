"""
FastAPI app for real-time house price prediction.

Run with:
    uvicorn deployment.api:app --reload --port 8000

Endpoints:
    POST /predict   -> predict SalePrice for a single house
    POST /reload-model -> force-refresh the cached model (e.g. after retraining)
    GET  /health    -> basic liveness/model-loaded check
"""

import logging

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from deployment.inference_features import engineer_features
from deployment.model_loader import get_model, reload_model
from deployment.schemas import HouseFeatures, PredictionResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = FastAPI(
    title="House Price Prediction API",
    description="Real-time inference for the prices_predictor model.",
    version="1.0.0",
)


@app.on_event("startup")
def _load_model_on_startup() -> None:
    try:
        _, version = get_model()
        logging.info(f"Model loaded successfully on startup (version: {version}).")
    except Exception as e:
        logging.error(
            f"Failed to load model on startup: {e}. "
            f"The API will start, but /predict will fail until a model "
            f"is available and /reload-model is called."
        )


@app.get("/health")
def health():
    try:
        _, version = get_model()
        return {"status": "ok", "model_version": version}
    except Exception as e:
        return {"status": "model_unavailable", "detail": str(e)}


@app.post("/reload-model")
def reload_model_endpoint():
    try:
        _, version = reload_model()
        return {"status": "reloaded", "model_version": version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {e}")


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    try:
        pipeline, model_version = get_model()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not available: {e}",
        )

    # by_alias=True so the resulting dict uses the original Ames Housing
    # column names ("Gr Liv Area", "Exter Qual", ...), matching what the
    # trained Pipeline's ColumnTransformer expects.
    raw_dict = features.model_dump(by_alias=True)
    raw_df = pd.DataFrame([raw_dict])

    try:
        engineered_df = engineer_features(raw_df)
        log_prediction = pipeline.predict(engineered_df)[0]
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # SalePrice was log1p-transformed during training (see
    # pipelines/training_pipeline.py, strategy="log", features=[...,
    # "SalePrice"]), so the model's raw output is in log-space. Invert
    # with expm1 to return an actual dollar amount.
    predicted_price = float(np.expm1(log_prediction))

    return PredictionResponse(
        predicted_sale_price=predicted_price,
        log_predicted_sale_price=float(log_prediction),
        model_version=model_version,
    )