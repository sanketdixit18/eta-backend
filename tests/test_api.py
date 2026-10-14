"""Backend tests. The real DagsHub model is replaced by a stub regressor, so these
verify the API, cleaning and the real preprocessor, but NOT the real model.
Run test_live_model.py (needs DAGSHUB_USER_TOKEN) for that.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.base import BaseEstimator, RegressorMixin

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app as backend  # noqa: E402

# Built from the field set the project's Streamlit app sends (same schema as the original Data model).
VALID = {
    "ID": "test-1",
    "Delivery_person_ID": "WEBRES00DEL00",
    "Delivery_person_Age": "30",
    "Delivery_person_Ratings": "4.6",
    "Restaurant_latitude": 19.1136,
    "Restaurant_longitude": 72.8697,
    "Delivery_location_latitude": 19.1176,
    "Delivery_location_longitude": 72.9047,
    "Order_Date": "14-10-2026",
    "Time_Orderd": "12:30:00",
    "Time_Order_picked": "12:40:00",
    "Weatherconditions": "conditions Sunny",
    "Road_traffic_density": "Medium",
    "Vehicle_condition": 2,
    "Type_of_order": "Meal",
    "Type_of_vehicle": "motorcycle",
    "multiple_deliveries": "1",
    "Festival": "No",
    "City": "Urban",
}


class StubRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.full(len(X), 31.5)


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(backend, "load_mlflow_model", lambda uri: StubRegressor())
    with TestClient(backend.app) as c:
        yield c


def test_files_present():
    for rel in ("app.py", "run_information.json", "scripts/data_clean_utils.py",
                "models/preprocessor.joblib", "requirements.txt", "Dockerfile"):
        assert (ROOT / rel).exists(), rel


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    assert r.json()["model_name"] == "model"


def test_predict_valid_returns_number(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200, r.text
    assert r.json() == {"prediction": 31.5}


def test_missing_field_is_422(client):
    bad = {k: v for k, v in VALID.items() if k != "Festival"}
    assert client.post("/predict", json=bad).status_code == 422


def test_wrong_type_is_422(client):
    assert client.post("/predict", json={**VALID, "Vehicle_condition": "abc"}).status_code == 422


def test_unparseable_date_is_422(client):
    r = client.post("/predict", json={**VALID, "Order_Date": "not-a-date"})
    assert r.status_code == 422


def test_minor_rider_is_422(client):
    r = client.post("/predict", json={**VALID, "Delivery_person_Age": "17"})
    assert r.status_code == 422


def test_route_over_25km_is_422(client):
    r = client.post("/predict", json={**VALID, "Delivery_location_latitude": 28.5355,
                                      "Delivery_location_longitude": 77.21})
    assert r.status_code == 422


def test_model_load_failure_gives_503(monkeypatch):
    def boom(uri):
        raise PermissionError("401 Unauthorized")

    monkeypatch.setattr(backend, "load_mlflow_model", boom)
    with TestClient(backend.app) as c:
        h = c.get("/health")
        assert h.status_code == 503
        assert "401 Unauthorized" in h.json()["detail"]
        assert c.post("/predict", json=VALID).status_code == 503


def test_real_preprocessor_runs_on_cleaned_row():
    import joblib
    from sklearn import set_config

    import pandas as pd
    from scripts.data_clean_utils import perform_data_cleaning

    set_config(transform_output="pandas")
    pre = joblib.load(ROOT / "models" / "preprocessor.joblib")
    cleaned = perform_data_cleaning(pd.DataFrame([VALID]))
    out = pre.transform(cleaned)
    assert len(out) == 1 and out.shape[1] > 5
