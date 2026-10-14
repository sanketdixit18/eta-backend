"""ETA Time Prediction - FastAPI backend.

The model is loaded once at startup from the MLflow registry on DagsHub; the
preprocessor is loaded from models/preprocessor.joblib. The prediction schema,
cleaning logic and response format are unchanged from the original project.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn import set_config
from sklearn.pipeline import Pipeline

from scripts.data_clean_utils import perform_data_cleaning

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("eta_backend")

BASE_DIR = Path(__file__).resolve().parent
RUN_INFO_PATH = BASE_DIR / "run_information.json"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.joblib"

# Same tracking server the original app.py used; override with an env var if needed.
DEFAULT_TRACKING_URI = "https://dagshub.com/sanketdixit18/ETA-Time-Prediction.mlflow"
DEFAULT_DAGSHUB_USER = "sanketdixit18"
# The original app.py hardcoded version "1" (run_information.json has no version).
DEFAULT_MODEL_VERSION = "1"

set_config(transform_output="pandas")


class Data(BaseModel):
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: str
    Delivery_person_Ratings: str
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: str
    Festival: str
    City: str


def configure_mlflow() -> str:
    """Point MLflow at DagsHub using env vars only (no secrets in code).

    DagsHub accepts the access token as both the MLflow username and password,
    or as username + token. We use DAGSHUB_USER_TOKEN (the name the dagshub
    client uses) or DAGSHUB_TOKEN, matching the original CI workflow.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    # Fail fast on auth/network problems instead of retrying for minutes at startup.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "60")
    mlflow.set_tracking_uri(tracking_uri)

    token = os.getenv("DAGSHUB_USER_TOKEN") or os.getenv("DAGSHUB_TOKEN")
    if token:
        os.environ.setdefault(
            "MLFLOW_TRACKING_USERNAME",
            os.getenv("DAGSHUB_USERNAME", DEFAULT_DAGSHUB_USER),
        )
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    elif not os.getenv("MLFLOW_TRACKING_PASSWORD"):
        logger.warning(
            "No DAGSHUB_USER_TOKEN / DAGSHUB_TOKEN set. Loading the model will "
            "fail if the DagsHub repository is private."
        )
    return tracking_uri


def load_mlflow_model(model_uri: str):
    return mlflow.sklearn.load_model(model_uri)


def build_pipeline() -> dict:
    """Load run info, preprocessor and MLflow model. Raises on any failure."""
    if not RUN_INFO_PATH.exists():
        raise FileNotFoundError(f"Model information file not found: {RUN_INFO_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor file not found: {PREPROCESSOR_PATH}")

    with open(RUN_INFO_PATH, "r", encoding="utf-8") as f:
        run_info = json.load(f)
    model_name = run_info["model_name"]
    model_version = os.getenv("MODEL_VERSION", DEFAULT_MODEL_VERSION)
    model_uri = f"models:/{model_name}/{model_version}"

    tracking_uri = configure_mlflow()
    logger.info("Loading model %s from %s", model_uri, tracking_uri)
    try:
        model = load_mlflow_model(model_uri)
    except Exception as exc:  # auth, network, missing version, incompatible pickle
        raise RuntimeError(
            f"Failed to load MLflow model '{model_name}' version '{model_version}' "
            f"({type(exc).__name__}: {exc}). Check the DagsHub token secret, "
            "the tracking URI and that the registered model version exists."
        ) from exc

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("regressor", model)])
    logger.info("Prediction pipeline ready.")
    return {
        "pipe": pipe,
        "model_name": model_name,
        "model_version": model_version,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipe = None
    app.state.load_error = None
    app.state.model_name = None
    app.state.model_version = None
    try:
        loaded = build_pipeline()
        app.state.pipe = loaded["pipe"]
        app.state.model_name = loaded["model_name"]
        app.state.model_version = loaded["model_version"]
    except Exception as exc:
        # Stay up so /health shows the reason instead of a crash loop in the Space logs.
        app.state.load_error = str(exc)
        logger.exception("Model failed to load; /predict will return 503.")
    yield


app = FastAPI(
    title="ETA Time Prediction",
    description="Delivery time prediction API using Machine Learning",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "ETA Time Prediction API",
        "health": "/health",
        "predict": "POST /predict",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    if app.state.pipe is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {app.state.load_error}",
        )
    return {
        "status": "healthy",
        "model_name": app.state.model_name,
        "model_version": app.state.model_version,
    }


@app.post("/predict")
def do_predictions(data: Data):
    if app.state.pipe is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Check /health and the server logs.",
        )

    pred_data = pd.DataFrame([data.model_dump()])

    try:
        cleaned_data = perform_data_cleaning(pred_data)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        logger.warning("Input cleaning failed: %s", exc)
        raise HTTPException(status_code=422, detail=f"Invalid input: {exc}") from exc

    if cleaned_data is None or cleaned_data.empty:
        raise HTTPException(
            status_code=422,
            detail=(
                "Input data cannot be used for prediction. "
                "Check rider age, route distance, coordinates, "
                "and order time against the supported data range."
            ),
        )

    try:
        prediction = app.state.pipe.predict(cleaned_data)[0]
    except (ValueError, TypeError, KeyError) as exc:
        logger.warning("Prediction input mismatch: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=f"Prediction input does not match the model: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected prediction error")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    return {"prediction": float(prediction)}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        workers=1,
    )
