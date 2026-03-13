import json
import logging
import pickle
from pathlib import Path
import os

import dagshub
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = Path("./models/model.pkl")
TEST_DATA_PATH = Path("./data/processed/test_bow.csv")
PARAMS_PATH = Path("params.yaml")

METRICS_PATH = Path("./reports/metrics.json")
MODEL_INFO_PATH = Path("./reports/experiment_info.json")

CONF_MATRIX_PATH = Path("confusion_matrix.png")

EXPERIMENT_NAME = "Logistic_regression_after_BOW"


# --------------------------------------------------
# MLflow + DagsHub Setup
# --------------------------------------------------

dagshub_token = os.getenv('DAGSHUB_CONN')
if not dagshub_token:
    raise EnvironmentError('DAGSHUB_CONN is not created. please set first')

os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

dagshub_url = 'https://dagshub.com'
repo_owner = 'Sudeep1245'
repo_name = "Mlops-mini-project"

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("model_evaluation")


# --------------------------------------------------
# Utility Functions
# --------------------------------------------------

def load_resources(model_path: Path, data_path: Path) -> tuple[BaseEstimator, pd.DataFrame]:
    """Load trained model and test dataset."""
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        data = pd.read_csv(data_path)

        logger.debug("Model and test data loaded successfully.")
        return model, data

    except Exception as e:
        logger.error("Failed loading resources: %s", e)
        raise


def split_features_labels(data: pd.DataFrame):
    """Split features and labels."""
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    logger.debug("Feature shape: %s | Label shape: %s", X.shape, y.shape)

    return X, y


def compute_metrics(y_true, y_pred, y_prob):
    """Compute evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob)
    }

    return metrics


def plot_confusion_matrix(y_true, y_pred):
    """Generate confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig(CONF_MATRIX_PATH)
    plt.close()

    logger.debug("Confusion matrix saved.")


def save_json(data: dict, path: Path):
    """Save dictionary as JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_params(path: Path):
    """Load YAML parameters."""
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------
# Main Evaluation Pipeline
# --------------------------------------------------

def evaluate_model():

    logger.debug("Starting model evaluation pipeline.")

    model, test_data = load_resources(MODEL_PATH, TEST_DATA_PATH)

    X_test, y_test = split_features_labels(test_data)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)

    plot_confusion_matrix(y_test, y_pred)

    params = load_params(PARAMS_PATH)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:

        logger.debug("MLflow run started.")

        mlflow.log_metrics(metrics)

        for section in params.values():
            for key, value in section.items():
                mlflow.log_param(key, value)

        mlflow.sklearn.log_model(model, 
                    artifact_path="Logistic_regression",
                    input_example= X_test, #automatically infers and logs a signature
                    registered_model_name='Logistic_regression')#To register a model

        mlflow.log_artifact(CONF_MATRIX_PATH)

        run_id = run.info.run_id

        save_json({"run_id": run_id, "model_path": "Logistic_regression"}, MODEL_INFO_PATH)

        logger.debug("Run ID: %s", run_id)
        logger.debug("Artifact URI: %s", mlflow.get_artifact_uri())

    save_json(metrics, METRICS_PATH)

    logger.debug("Evaluation completed successfully.")


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":

    logger.debug("Tracking URI: %s", mlflow.get_tracking_uri())

    evaluate_model()