"""
Loads the latest trained sklearn Pipeline produced by model_building_step,
via ZenML's Client/Model API, so the FastAPI app always serves whatever
model was most recently registered under the "prices_predictor" model
name — matching the Model(name="prices_predictor", ...) used in
steps/model_building.py and pipelines/training_pipeline.py.

The loaded pipeline is cached in-process after the first successful load.
Call reload_model() to force a refresh (e.g. after retraining) without
restarting the API process.
"""

import logging
import threading

from sklearn.pipeline import Pipeline
from zenml.client import Client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

MODEL_NAME = "prices_predictor"
ARTIFACT_NAME = "sklearn_pipeline"

_model_lock = threading.Lock()
_cached_pipeline: Pipeline | None = None
_cached_model_version: str | None = None


def _fetch_latest_pipeline() -> tuple[Pipeline, str]:
    client = Client()

    model_version = client.get_model_version(
        model_name_or_id=MODEL_NAME,
        model_version_name_or_number_or_id=None,
    )

    artifact = model_version.get_artifact(ARTIFACT_NAME)
    if artifact is None:
        raise RuntimeError(
            f"No artifact named '{ARTIFACT_NAME}' found on model "
            f"'{MODEL_NAME}' version '{model_version.name}'. "
            f"Has the training pipeline been run at least once?"
        )

    pipeline: Pipeline = artifact.load()
    logging.info(
        f"Loaded '{ARTIFACT_NAME}' from model '{MODEL_NAME}' "
        f"version '{model_version.name}'."
    )
    return pipeline, model_version.name


def get_model() -> tuple[Pipeline, str]:
    global _cached_pipeline, _cached_model_version

    if _cached_pipeline is None:
        with _model_lock:
            if _cached_pipeline is None:  
                _cached_pipeline, _cached_model_version = _fetch_latest_pipeline()

    return _cached_pipeline, _cached_model_version


def reload_model() -> tuple[Pipeline, str]:
    global _cached_pipeline, _cached_model_version

    with _model_lock:
        _cached_pipeline, _cached_model_version = _fetch_latest_pipeline()

    return _cached_pipeline, _cached_model_version