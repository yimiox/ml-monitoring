"""
train.py
Trains a GradientBoosting regressor on the house-price reference dataset,
logs everything to MLflow, and registers the model in the Model Registry.

Run inside Docker (after `docker compose up mlflow -d`):
    docker compose run --rm monitor python scripts/train.py

Or locally (with MLflow server running on localhost:5000):
    MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/train.py
"""

import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Config ────────────────────────────────────────────────────────────────
TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "house_price_monitor")
MODEL_NAME      = os.getenv("MLFLOW_MODEL_NAME",      "house_price_model")
DATA_PATH       = Path(os.getenv("REFERENCE_DATA_PATH", "data/reference/reference.csv"))
MODEL_OUT       = Path("models/model.pkl")

FEATURES = ["sqft", "bedrooms", "bathrooms", "age_years",
            "garage", "neighbourhood", "distance_cbd"]
TARGET   = "price_usd"

CAT_FEATURES = ["neighbourhood"]
NUM_FEATURES = [f for f in FEATURES if f not in CAT_FEATURES]

# ── Model hyperparameters (logged to MLflow) ──────────────────────────────
PARAMS = {
    "n_estimators":    200,
    "max_depth":       4,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "random_state":    42,
}


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         CAT_FEATURES),
    ], remainder="passthrough")   # numeric features pass through unchanged

    model = GradientBoostingRegressor(**PARAMS)
    return Pipeline([("prep", preprocessor), ("model", model)])


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "mae":  mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2":   r2_score(y_true, y_pred),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def train():
    # ── Load data ─────────────────────────────────────────────────────────
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURES]
    y  = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── MLflow setup ──────────────────────────────────────────────────────
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="baseline_gbr") as run:
        print(f"\nMLflow run: {run.info.run_id}")

        # ── Train ─────────────────────────────────────────────────────────
        pipe = build_pipeline()
        pipe.fit(X_train, y_train)

        # ── Evaluate ──────────────────────────────────────────────────────
        train_metrics = compute_metrics(y_train, pipe.predict(X_train))
        test_metrics  = compute_metrics(y_test,  pipe.predict(X_test))

        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")

        print("\n── Test metrics ──────────────────────────────")
        for k, v in test_metrics.items():
            print(f"  {k:6s}: {v:.4f}")
        print(f"  CV R²  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # ── Log to MLflow ─────────────────────────────────────────────────
        mlflow.log_params(PARAMS)
        mlflow.log_params({"features": str(FEATURES), "target": TARGET})

        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.log_metric("cv_r2_mean", cv_scores.mean())
        mlflow.log_metric("cv_r2_std",  cv_scores.std())

        # Feature metadata artifact — Evidently needs this in Phase 3
        feature_meta = {
            "features":     FEATURES,
            "cat_features": CAT_FEATURES,
            "num_features": NUM_FEATURES,
            "target":       TARGET,
        }
        meta_path = Path("models/feature_meta.json")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(feature_meta, indent=2))
        mlflow.log_artifact(str(meta_path), artifact_path="metadata")

        # Log the sklearn pipeline via MLflow's native sklearn flavour
        signature = mlflow.models.infer_signature(X_train, pipe.predict(X_train))
        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            signature=signature,
            registered_model_name=MODEL_NAME,
        )

        # Local pickle for fast loading in the monitoring service
        MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_OUT, "wb") as f:
            pickle.dump(pipe, f)

        run_id = run.info.run_id

    # ── Promote to Staging ────────────────────────────────────────────────
    client = MlflowClient(tracking_uri=TRACKING_URI)
    versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
    if versions:
        latest = versions[0]
        client.transition_model_version_stage(
            name=MODEL_NAME, version=latest.version, stage="Staging",
        )
        print(f"\n✓ Model v{latest.version} promoted → Staging")

    print(f"✓ Local model saved : {MODEL_OUT}")
    print(f"\nDone. Run ID: {run_id}")
    return run_id


if __name__ == "__main__":
    train()
