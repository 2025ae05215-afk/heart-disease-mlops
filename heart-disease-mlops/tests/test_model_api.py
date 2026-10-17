import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import api
from src.data import clean, split_xy
from src.features import build_preprocessor
from tests.test_data import make_df

SAMPLE = {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
          "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
          "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}


@pytest.fixture(scope="module")
def client():
    X, y = split_xy(clean(make_df(80)))
    y = (X["age"] > 50).astype(int)  # deterministic learnable target
    api._model = Pipeline([("prep", build_preprocessor()),
                           ("clf", LogisticRegression(max_iter=500))]).fit(X, y)
    return TestClient(api.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_ok(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0


def test_predict_validation_error(client):
    bad = {**SAMPLE, "age": -5}
    assert client.post("/predict", json=bad).status_code == 422
    assert client.post("/predict", json={"age": 40}).status_code == 422


def test_metrics_endpoint(client):
    client.post("/predict", json=SAMPLE)
    text = client.get("/metrics").text
    assert "api_requests_total" in text and "predictions_total" in text
