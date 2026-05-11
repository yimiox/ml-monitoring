"""
monitor.py
Runs on a schedule, compares the latest incoming data batch against the
reference dataset using Evidently, then exposes drift scores as Prometheus
metrics on :8000/metrics.

Three things are measured every cycle:
  1. Data drift   — have the input feature distributions shifted?
  2. Target drift — has the distribution of predictions shifted?
  3. Model quality— has MAPE / R² degraded (when ground truth is available)?
"""

import os
import time
import pickle
import logging
from pathlib import Path

import pandas as pd
from prometheus_client import Gauge, Counter, start_http_server

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DataDriftTable,
    ColumnDriftMetric,
)

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────
REFERENCE_PATH  = Path(os.getenv("REFERENCE_DATA_PATH", "/data/reference/reference.csv"))
INCOMING_DIR    = Path(os.getenv("INCOMING_DATA_PATH",  "/data/incoming"))
MODEL_PATH      = Path(os.getenv("MODEL_PATH",          "/models/model.pkl"))
METRICS_PORT    = int(os.getenv("METRICS_PORT",         "8000"))
CHECK_INTERVAL  = int(os.getenv("DRIFT_CHECK_INTERVAL", "30"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD",    "0.15"))

FEATURES = ["sqft", "bedrooms", "bathrooms", "age_years",
            "garage", "neighbourhood", "distance_cbd"]
TARGET   = "price_usd"

# ── Prometheus metrics ────────────────────────────────────────────────────
DATASET_DRIFT_SCORE    = Gauge("ml_dataset_drift_score",    "Share of features drifted (0-1)")
DATASET_DRIFT_DETECTED = Gauge("ml_dataset_drift_detected", "1 if dataset drift detected")
FEATURE_DRIFT_SCORE    = Gauge("ml_feature_drift_score",    "Per-feature drift score", ["feature"])
FEATURE_DRIFT_DETECTED = Gauge("ml_feature_drift_detected", "1 if feature drifted",    ["feature"])
PREDICTION_DRIFT_SCORE = Gauge("ml_prediction_drift_score", "Drift score on predictions")
MODEL_MAE              = Gauge("ml_model_mae",  "Mean Absolute Error on current batch")
MODEL_RMSE             = Gauge("ml_model_rmse", "Root Mean Squared Error on current batch")
MODEL_R2               = Gauge("ml_model_r2",   "R² score on current batch")
MODEL_MAPE             = Gauge("ml_model_mape", "Mean Absolute Percentage Error (%)")
CHECKS_TOTAL           = Counter("ml_monitor_checks_total",  "Total drift checks run")
CHECKS_FAILED          = Counter("ml_monitor_checks_failed", "Drift checks that errored")
DRIFT_ALERTS           = Counter("ml_drift_alerts_total",    "Times drift threshold breached")
BATCH_SIZE             = Gauge("ml_current_batch_size",      "Rows in latest batch")
LAST_CHECK_TS          = Gauge("ml_last_check_timestamp",    "Unix ts of last check")


# ── Helpers ───────────────────────────────────────────────────────────────
def load_model():
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    log.warning("model.pkl not found — prediction metrics skipped")
    return None


def load_reference() -> pd.DataFrame:
    df = pd.read_csv(REFERENCE_PATH)
    log.info(f"Reference loaded: {len(df):,} rows")
    return df


def load_latest_batch():
    csvs = sorted(INCOMING_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not csvs:
        log.warning("No incoming batch files — skipping")
        return None
    df = pd.read_csv(csvs[-1])
    log.info(f"Batch: {csvs[-1].name} ({len(df):,} rows)")
    return df


def add_predictions(df: pd.DataFrame, model) -> pd.DataFrame:
    df = df.copy()
    df["prediction"] = model.predict(df[FEATURES])
    return df


# ── Drift checks ──────────────────────────────────────────────────────────
def run_data_drift(reference: pd.DataFrame, current: pd.DataFrame):
    """
    Runs KS-test (numeric) and chi-squared (categorical) on every feature.
    DatasetDriftMetric gives an overall flag; ColumnDriftMetric per feature.
    """
    metrics_list = [DatasetDriftMetric(), DataDriftTable()]
    for feat in FEATURES:
        metrics_list.append(ColumnDriftMetric(column_name=feat))

    report = Report(metrics=metrics_list)
    report.run(reference_data=reference[FEATURES], current_data=current[FEATURES])
    result = report.as_dict()["metrics"]

    # Dataset level
    ds = result[0]["result"]
    drift_share    = ds["share_of_drifted_columns"]
    drift_detected = int(ds["dataset_drift"])
    DATASET_DRIFT_SCORE.set(drift_share)
    DATASET_DRIFT_DETECTED.set(drift_detected)

    if drift_detected:
        DRIFT_ALERTS.inc()
        log.warning(f"DRIFT DETECTED — {drift_share:.1%} of features drifted")
    else:
        log.info(f"No drift — share: {drift_share:.1%}")

    # Per-feature (index 2 onwards)
    for i, feat in enumerate(FEATURES):
        col = result[2 + i]["result"]
        score    = col.get("drift_score", 0.0)
        detected = int(col.get("drift_detected", False))
        FEATURE_DRIFT_SCORE.labels(feature=feat).set(score)
        FEATURE_DRIFT_DETECTED.labels(feature=feat).set(detected)
        log.info(f"  {feat:20s} score={score:.4f}  {'⚠ DRIFT' if detected else 'ok'}")

    return drift_detected


def run_prediction_drift(reference: pd.DataFrame, current: pd.DataFrame, model):
    """
    Detects concept drift by checking if the prediction distribution has shifted.
    Useful when you don't have ground-truth labels yet.
    """
    ref_p = add_predictions(reference, model)
    cur_p = add_predictions(current,   model)

    report = Report(metrics=[ColumnDriftMetric(column_name="prediction")])
    report.run(
        reference_data=ref_p[["prediction"]],
        current_data=cur_p[["prediction"]],
    )
    score = report.as_dict()["metrics"][0]["result"].get("drift_score", 0.0)
    PREDICTION_DRIFT_SCORE.set(score)
    log.info(f"  {'prediction':20s} score={score:.4f}")
    return score


def run_regression_quality(reference: pd.DataFrame, current: pd.DataFrame, model):
    """
    Computes MAE/RMSE/R²/MAPE on the current batch vs reference.
    Only runs when ground-truth labels exist in the batch.
    """
    if TARGET not in current.columns:
        log.info("No ground-truth labels — skipping regression metrics")
        return

    ref_p = add_predictions(reference, model)
    cur_p = add_predictions(current,   model)

    from evidently.utils.data_operations import DataColumnMapping
    report = Report(metrics=[RegressionPreset()])
    report.run(
        reference_data=ref_p[FEATURES + [TARGET, "prediction"]],
        current_data=cur_p[FEATURES   + [TARGET, "prediction"]],
        column_mapping=DataColumnMapping(target=TARGET, prediction="prediction"),
    )

    for m in report.as_dict()["metrics"]:
        curr = m.get("result", {}).get("current", {})
        if "mean_abs_error"       in curr: MODEL_MAE.set(curr["mean_abs_error"]);        log.info(f"  MAE  = {curr['mean_abs_error']:,.0f}")
        if "mean_sq_error"        in curr: MODEL_RMSE.set(curr["mean_sq_error"] ** 0.5)
        if "r2_score"             in curr: MODEL_R2.set(curr["r2_score"]);               log.info(f"  R²   = {curr['r2_score']:.4f}")
        if "mean_abs_perc_error"  in curr: MODEL_MAPE.set(curr["mean_abs_perc_error"]);  log.info(f"  MAPE = {curr['mean_abs_perc_error']:.2f}%")


# ── Main loop ─────────────────────────────────────────────────────────────
def run_check(reference: pd.DataFrame, model):
    CHECKS_TOTAL.inc()
    try:
        current = load_latest_batch()
        if current is None:
            return
        BATCH_SIZE.set(len(current))

        log.info("── Data drift ────────────────────────────────")
        run_data_drift(reference, current)

        if model:
            log.info("── Prediction drift ──────────────────────────")
            run_prediction_drift(reference, current, model)
            log.info("── Regression quality ────────────────────────")
            run_regression_quality(reference, current, model)

        LAST_CHECK_TS.set(time.time())

    except Exception as e:
        CHECKS_FAILED.inc()
        log.error(f"Check failed: {e}", exc_info=True)


def main():
    log.info(f"Starting ML Monitor — metrics on :{METRICS_PORT}/metrics")
    start_http_server(METRICS_PORT)
    log.info("Prometheus metrics server started")

    reference = load_reference()
    model     = load_model()

    log.info("Running first check...")
    run_check(reference, model)

    while True:
        time.sleep(CHECK_INTERVAL)
        log.info("── Starting drift check ──────────────────────")
        run_check(reference, model)


if __name__ == "__main__":
    main()