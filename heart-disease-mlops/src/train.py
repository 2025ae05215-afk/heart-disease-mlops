"""Train LR + RF with CV & grid search, log everything to MLflow, save best."""
import json
import tempfile
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,  # noqa: E402
                             accuracy_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,  # noqa: E402
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline  # noqa: E402

from .config import (ARTIFACT_ROOT, EXPERIMENT, METRICS_PATH, MLFLOW_URI, MODEL_DIR,  # noqa: E402
                     MODEL_PATH, RANDOM_STATE)
from .data import clean, load_raw, split_xy  # noqa: E402
from .features import build_preprocessor  # noqa: E402

CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        {"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l2"]}),
    "random_forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"clf__n_estimators": [100, 300], "clf__max_depth": [3, 5, None],
         "clf__min_samples_leaf": [1, 3]}),
}
SCORING = {"accuracy": "accuracy", "precision": "precision",
           "recall": "recall", "roc_auc": "roc_auc"}


def evaluate(pipe, X, y) -> dict:
    pred, proba = pipe.predict(X), pipe.predict_proba(X)[:, 1]
    return {"accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred),
            "recall": recall_score(y, pred),
            "roc_auc": roc_auc_score(y, proba)}


def run() -> dict:
    mlflow.set_tracking_uri(MLFLOW_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_ROOT)
    mlflow.set_experiment(EXPERIMENT)
    X, y = split_xy(clean(load_raw()))
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    best_name, best_auc, best_pipe, summary = None, -1, None, {}
    for name, (clf, grid) in CANDIDATES.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        with mlflow.start_run(run_name=name):
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
            gs.fit(Xtr, ytr)
            est = gs.best_estimator_
            cvres = cross_validate(est, Xtr, ytr, cv=cv, scoring=SCORING)
            cv_m = {f"cv_{k}": float(cvres[f"test_{k}"].mean()) for k in SCORING}
            test_m = {f"test_{k}": v for k, v in evaluate(est, Xte, yte).items()}

            mlflow.log_params({k.replace("clf__", ""): v
                               for k, v in gs.best_params_.items()})
            mlflow.log_param("model", name)
            mlflow.log_metrics({**cv_m, **test_m})
            with tempfile.TemporaryDirectory() as tmp:
                fig, ax = plt.subplots()
                RocCurveDisplay.from_estimator(est, Xte, yte, ax=ax)
                fig.savefig(Path(tmp) / "roc.png")
                plt.close(fig)
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay.from_estimator(est, Xte, yte, ax=ax)
                fig.savefig(Path(tmp) / "confusion.png")
                plt.close(fig)
                mlflow.log_artifacts(tmp, artifact_path="plots")
            mlflow.sklearn.log_model(est, artifact_path="model",
                                     serialization_format="cloudpickle")

            summary[name] = {**cv_m, **test_m, "params": gs.best_params_}
            if cv_m["cv_roc_auc"] > best_auc:
                best_name, best_auc, best_pipe = name, cv_m["cv_roc_auc"], est
            print(f"{name}: cv_auc={cv_m['cv_roc_auc']:.3f} "
                  f"test_auc={test_m['test_roc_auc']:.3f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_pipe, MODEL_PATH)  # preprocessing + model in ONE artifact
    METRICS_PATH.write_text(json.dumps(
        {"best_model": best_name, "results": summary}, indent=2, default=str))
    print(f"best={best_name} -> {MODEL_PATH}")
    return summary


if __name__ == "__main__":
    run()
