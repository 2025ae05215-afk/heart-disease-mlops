# Heart Disease Risk - End-to-End MLOps

Pipeline: data -> EDA -> train (LR + RF, MLflow) -> packaged pipeline -> FastAPI -> Docker -> Kubernetes -> Prometheus/Grafana, with GitHub Actions CI.

## Setup (clean machine)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py     # UCI first, mirror fallback -> data/heart_raw.csv
# or use a local UCI file: python scripts/download_data.py path/to/processed.cleveland.data
python -m src.eda                   # plots -> screenshots/
python -m src.train                 # MLflow runs + models/model.joblib
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
pytest -v && flake8 src tests scripts
```

## Serve locally
```bash
uvicorn src.api:app --port 8000
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
 -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```
Endpoints: `POST /predict`, `GET /health`, `GET /metrics`.

## Docker
```bash
docker build -t heart-api:latest .
docker run -p 8000:8000 heart-api:latest
```

## Kubernetes (Minikube)
```bash
minikube start
eval $(minikube docker-env) && docker build -t heart-api:latest .   # image visible to cluster
kubectl apply -f k8s/
kubectl get pods,svc
minikube tunnel                      # separate terminal; gives LoadBalancer EXTERNAL-IP
curl http://<EXTERNAL-IP>/health
```
(Docker Desktop K8s: skip the minikube lines; LoadBalancer is on localhost.)

## Monitoring
```bash
docker compose up --build            # api :8000, Prometheus :9090, Grafana :3000
```
Grafana (admin/admin) -> Prometheus datasource is pre-provisioned. Example panels:
- Request rate: `sum(rate(api_requests_total[1m])) by (path)`
- p95 latency: `histogram_quantile(0.95, sum(rate(api_request_latency_seconds_bucket[5m])) by (le))`
- Predictions by class: `sum(predictions_total) by (label)`
- Error rate: `sum(rate(api_requests_total{status=~"5.."}[1m]))`

## Layout
`src/` code | `tests/` pytest | `scripts/` data download | `notebooks/` EDA, training, inference | `k8s/` manifests | `monitoring/` Prometheus/Grafana | `.github/workflows/ci.yml` | `screenshots/` report images
