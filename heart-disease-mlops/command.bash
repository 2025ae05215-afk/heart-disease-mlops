Since you’re already signed in to GitHub in VS Code, the easiest route is to clone your pushed repo.

1. Open the repo in VS Code

Press Ctrl+Shift+P and choose Git: Clone.
Choose Clone from GitHub, select heart-disease-mlops, pick a folder, then click Open.
Open the terminal with Ctrl+`.

If the repo doesn’t appear in the list, paste its URL instead. Without Git in the VM, use the zip: upload it, unzip it, then use File > Open Folder.

2. Set up Python (if python3 is missing, ask the lab admin or install Python 3.11)

Linux VM:

bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt



3. Run the pipeline (the data file is already in the repo)

bash
python scripts/download_data.py data/processed.cleveland.data
python -m src.eda
python -m src.train
pytest -v
flake8 src tests scripts

Compare the cv_auc and test_auc lines in the training output with the report. On the UCI file I got Logistic Regression 0.902 / 0.966 and Random Forest 0.901 / 0.947. Screenshot the training output, the passing tests and the MLflow UI.

4. MLflow UI (screenshot at http://localhost:5000)

bash
mlflow ui --backend-store-uri sqlite:///mlflow.db

5. Run the API locally (in a second terminal, with the venv activated)

bash
uvicorn src.api:app --port 8000

Open http://localhost:8000/docs and use the /predict form, or from another terminal:

bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d "{\"age\":63,\"sex\":1,\"cp\":1,\"trestbps\":145,\"chol\":233,\"fbs\":1,\"restecg\":2,\"thalach\":150,\"exang\":0,\"oldpeak\":2.3,\"slope\":3,\"ca\":0,\"thal\":6}"

6. Docker and Kubernetes

bash
docker build -t heart-api:latest .
docker run -p 8000:8000 heart-api:latest

For Minikube:

bash
minikube start
eval $(minikube docker-env)       # Linux VM; Windows cmd: @FOR /f "tokens=*" %i IN ('minikube -p minikube docker-env --shell cmd') DO @%i
docker build -t heart-api:latest .
kubectl apply -f k8s/
kubectl get pods,svc
minikube tunnel                   # keep running in a separate terminal

Then call /health and /predict on the service’s EXTERNAL-IP.

For monitoring:

bash
docker compose up --build

Then open Prometheus at localhost:9090 and Grafana at localhost:3000 (admin / admin). Send a few /predict requests first so the panels have data.

If something fails

docker: permission denied on Linux: run sudo usermod -aG docker $USER, then log out and back in.
pip can’t reach the internet: the lab may need a proxy or a local mirror, so ask the admin.
The model file is needed by the Docker build: models/model.joblib is created by python -m src.train. If it’s missing, train first.
