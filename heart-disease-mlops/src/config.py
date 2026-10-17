"""Central configuration: paths, column names, constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "heart_raw.csv"
CLEAN_PATH = DATA_DIR / "heart_clean.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
ARTIFACT_ROOT = (ROOT / "mlartifacts").as_uri()
EXPERIMENT = "heart-disease-risk"

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "target"
RANDOM_STATE = 42
