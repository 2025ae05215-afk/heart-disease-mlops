"""FastAPI serving app: /predict, /health, /metrics (Prometheus), request logs."""
import logging
import time

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from pydantic import BaseModel, Field

from .config import FEATURES, MODEL_PATH

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("heart-api")

REQS = Counter("api_requests_total", "Requests", ["path", "status"])
LAT = Histogram("api_request_latency_seconds", "Latency", ["path"])
PREDS = Counter("predictions_total", "Predictions by class", ["label"])

app = FastAPI(title="Heart Disease Risk API", version="1.0.0")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


class Patient(BaseModel):
    """Ranges are permissive: they accept both UCI and mirror category codings."""
    age: float = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=4)
    trestbps: float = Field(..., ge=50, le=260)
    chol: float = Field(..., ge=80, le=700)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=40, le=260)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=0, le=3)
    ca: int = Field(..., ge=0, le=4)
    thal: int = Field(..., ge=0, le=7)


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    dt = time.perf_counter() - start
    path = request.url.path
    REQS.labels(path, response.status_code).inc()
    LAT.labels(path).observe(dt)
    log.info("method=%s path=%s status=%s latency_ms=%.1f",
             request.method, path, response.status_code, dt * 1000)
    return response


@app.get("/health")
def health():
    get_model()
    return {"status": "ok"}


@app.post("/predict")
def predict(p: Patient):
    row = pd.DataFrame([p.model_dump()])[FEATURES]
    proba = float(get_model().predict_proba(row)[0, 1])
    label = int(proba >= 0.5)
    PREDS.labels(str(label)).inc()
    log.info("prediction=%d confidence=%.3f", label, proba)
    return {"prediction": label,
            "confidence": round(proba if label else 1 - proba, 4),
            "probability_disease": round(proba, 4)}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
