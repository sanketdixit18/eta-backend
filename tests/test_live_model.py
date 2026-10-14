"""Optional live check against the real DagsHub model. Skipped unless a token is set."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytestmark = pytest.mark.skipif(
    not (os.getenv("DAGSHUB_USER_TOKEN") or os.getenv("DAGSHUB_TOKEN")),
    reason="DAGSHUB_USER_TOKEN not set",
)


def test_real_model_predicts():
    from fastapi.testclient import TestClient

    import app as backend
    from tests.test_api import VALID

    with TestClient(backend.app) as c:
        h = c.get("/health")
        assert h.status_code == 200, h.text
        r = c.post("/predict", json=VALID)
        assert r.status_code == 200, r.text
        assert isinstance(r.json()["prediction"], float)
