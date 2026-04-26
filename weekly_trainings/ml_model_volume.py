"""
Standalone weekly training script for the CI San Francisco volume model.

This script does NOT import app.py. It:
1. connects to PostgreSQL directly using DB_* env vars
2. builds the ML dataset from risk_features_hourly + incident_counts_hourly
3. trains the calibrated two-stage volume model
4. writes local artifacts
5. uploads .joblib and metrics JSON to Railway Bucket / S3-compatible storage

Usage:
    python ml_model_volume.py

Required env vars:
    DB_HOST
    DB_NAME
    DB_USER
    DB_PASSWORD
    DB_PORT

Required for bucket upload:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION
    AWS_ENDPOINT_URL
    AWS_S3_BUCKET_NAME

Optional env vars:
    ML_LOOKBACK_DAYS=180
    ML_TEST_SIZE=0.2
    ML_MIN_ROWS=200
    MODEL_DIR=models
    VOLUME_MODEL_NAME=volume_random_forest_v1
    MODEL_S3_PREFIX=models
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "models"))
VOLUME_MODEL_NAME = os.environ.get("VOLUME_MODEL_NAME", "volume_random_forest_v1")
DEFAULT_ML_LOOKBACK_DAYS = int(os.environ.get("ML_LOOKBACK_DAYS", "180"))
DEFAULT_ML_TEST_SIZE = float(os.environ.get("ML_TEST_SIZE", "0.2"))
DEFAULT_ML_MIN_ROWS = int(os.environ.get("ML_MIN_ROWS", "200"))

MODEL_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME")
MODEL_BUCKET_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL")
MODEL_BUCKET_REGION = os.environ.get("AWS_DEFAULT_REGION", "auto")
MODEL_S3_PREFIX = os.environ.get("MODEL_S3_PREFIX", "models")

ML_NUMERIC_FEATURES = [
    "hour_of_day",
    "month_of_year",
    "incidents_last_1h",
    "incidents_last_3h",
    "incidents_last_6h",
    "incidents_last_24h",
    "incidents_last_7d",
    "open_active_ratio_24h",
    "filed_online_ratio_24h",
    "avg_report_delay_minutes_24h",
]

ML_CATEGORICAL_FEATURES = [
    "police_district",
    "incident_category",
    "day_of_week",
]

ML_TARGET_COLUMN = "target_incidents_next_hour"


def parse_int_env(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.environ.get(name)
    try:
        value = int(raw) if raw is not None else default
    except ValueError:
        value = default
    return max(min_value, min(value, max_value))


def parse_float_env(name: str, default: float, min_value: float, max_value: float) -> float:
    raw = os.environ.get(name)
    try:
        value = float(raw) if raw is not None else default
    except ValueError:
        value = default
    return max(min_value, min(value, max_value))


def get_db_connection() -> Any:
    required = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_PORT"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing required DB env vars: {missing}")

    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        database=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        port=os.environ["DB_PORT"],
    )


def fetch_all_dict(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]


def fetch_volume_ml_dataset(lookback_days: int) -> list[dict[str, Any]]:
    return fetch_all_dict(
        """
        WITH next_counts AS (
            SELECT
                bucket_start,
                police_district,
                incident_category,
                SUM(total_incidents) AS total_incidents
            FROM incident_counts_hourly
            GROUP BY 1,2,3
        ), bounds AS (
            SELECT MAX(bucket_start) AS max_bucket_start
            FROM incident_counts_hourly
        )
        SELECT
            f.feature_timestamp,
            f.police_district,
            f.incident_category,
            f.hour_of_day,
            f.day_of_week,
            f.month_of_year,
            COALESCE(f.incidents_last_1h, 0) AS incidents_last_1h,
            COALESCE(f.incidents_last_3h, 0) AS incidents_last_3h,
            COALESCE(f.incidents_last_6h, 0) AS incidents_last_6h,
            COALESCE(f.incidents_last_24h, 0) AS incidents_last_24h,
            COALESCE(f.incidents_last_7d, 0) AS incidents_last_7d,
            COALESCE(f.open_active_ratio_24h, 0) AS open_active_ratio_24h,
            COALESCE(f.filed_online_ratio_24h, 0) AS filed_online_ratio_24h,
            COALESCE(f.avg_report_delay_minutes_24h, 0) AS avg_report_delay_minutes_24h,
            COALESCE(n.total_incidents, 0) AS target_incidents_next_hour
        FROM risk_features_hourly f
        CROSS JOIN bounds b
        LEFT JOIN next_counts n
            ON n.bucket_start = f.feature_timestamp + INTERVAL '1 hour'
           AND n.police_district = f.police_district
           AND n.incident_category = f.incident_category
        WHERE f.feature_timestamp >= NOW() - (%s::int * INTERVAL '1 day')
          AND f.feature_timestamp < b.max_bucket_start
          AND COALESCE(f.police_district, '') <> ''
          AND COALESCE(f.incident_category, '') <> ''
        ORDER BY f.feature_timestamp ASC, f.police_district ASC, f.incident_category ASC;
        """,
        (lookback_days,),
    )


def summarize_volume_ml_dataset(rows: list[dict[str, Any]], lookback_days: int) -> dict[str, Any]:
    if not rows:
        return {
            "lookback_days": lookback_days,
            "row_count": 0,
            "min_feature_timestamp": None,
            "max_feature_timestamp": None,
            "district_count": 0,
            "category_count": 0,
            "target_sum": 0,
            "target_avg": 0,
            "target_max": 0,
        }

    timestamps = [row["feature_timestamp"] for row in rows if row.get("feature_timestamp")]
    targets = [float(row.get(ML_TARGET_COLUMN) or 0) for row in rows]
    districts = {row.get("police_district") for row in rows if row.get("police_district")}
    categories = {row.get("incident_category") for row in rows if row.get("incident_category")}

    return {
        "lookback_days": lookback_days,
        "row_count": len(rows),
        "min_feature_timestamp": min(timestamps).isoformat() if timestamps else None,
        "max_feature_timestamp": max(timestamps).isoformat() if timestamps else None,
        "district_count": len(districts),
        "category_count": len(categories),
        "target_sum": round(sum(targets), 4),
        "target_avg": round(sum(targets) / len(targets), 4) if targets else 0,
        "target_max": round(max(targets), 4) if targets else 0,
    }


def get_model_artifact_keys() -> dict[str, str]:
    prefix = MODEL_S3_PREFIX.rstrip("/")
    return {
        "model": f"{prefix}/{VOLUME_MODEL_NAME}.joblib",
        "metrics": f"{prefix}/{VOLUME_MODEL_NAME}_metrics.json",
    }


def get_s3_client() -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Missing S3 dependency. Add boto3 to requirements.txt.") from exc

    if not MODEL_BUCKET_NAME:
        raise RuntimeError("AWS_S3_BUCKET_NAME is not configured.")
    if not MODEL_BUCKET_ENDPOINT_URL:
        raise RuntimeError("AWS_ENDPOINT_URL is not configured.")

    return boto3.client(
        "s3",
        endpoint_url=MODEL_BUCKET_ENDPOINT_URL,
        region_name=MODEL_BUCKET_REGION,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def upload_bytes_to_model_bucket(key: str, data: bytes, content_type: str) -> None:
    client = get_s3_client()
    client.put_object(
        Bucket=MODEL_BUCKET_NAME,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def upload_volume_model_artifacts(model_path: Path, metrics_path: Path) -> dict[str, Any]:
    keys = get_model_artifact_keys()

    upload_bytes_to_model_bucket(
        key=keys["model"],
        data=model_path.read_bytes(),
        content_type="application/octet-stream",
    )
    upload_bytes_to_model_bucket(
        key=keys["metrics"],
        data=metrics_path.read_bytes(),
        content_type="application/json",
    )

    return {
        "bucket": MODEL_BUCKET_NAME,
        "endpoint_url": MODEL_BUCKET_ENDPOINT_URL,
        "model_key": keys["model"],
        "metrics_key": keys["metrics"],
    }


def make_preprocessor() -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("categorical", encoder, ML_CATEGORICAL_FEATURES),
            ("numeric", "passthrough", ML_NUMERIC_FEATURES),
        ]
    )


def train_volume_forecast_model(rows: list[dict[str, Any]], test_size: float) -> dict[str, Any]:
    if len(rows) < DEFAULT_ML_MIN_ROWS:
        raise ValueError(f"Not enough rows to train. Required at least {DEFAULT_ML_MIN_ROWS}, got {len(rows)}.")

    df = pd.DataFrame(rows).sort_values("feature_timestamp").reset_index(drop=True)

    for column in ML_NUMERIC_FEATURES + [ML_TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    for column in ML_CATEGORICAL_FEATURES:
        df[column] = df[column].fillna("Unknown").astype(str)

    df["target_has_incident_next_hour"] = (df[ML_TARGET_COLUMN] > 0).astype(int)

    feature_columns = ML_NUMERIC_FEATURES + ML_CATEGORICAL_FEATURES
    X = df[feature_columns]
    y_count = df[ML_TARGET_COLUMN]
    y_binary = df["target_has_incident_next_hour"]

    split_index = int(len(df) * (1.0 - test_size))
    split_index = max(1, min(split_index, len(df) - 1))

    X_train = X.iloc[:split_index]
    y_count_train = y_count.iloc[:split_index]
    y_binary_train = y_binary.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_count_test = y_count.iloc[split_index:]
    y_binary_test = y_binary.iloc[split_index:]

    positive_train_mask = y_count_train > 0
    positive_train_rows = int(positive_train_mask.sum())
    if positive_train_rows < max(20, int(DEFAULT_ML_MIN_ROWS * 0.05)):
        raise ValueError(
            "Not enough positive target rows to train the second-stage regressor. "
            f"Positive train rows: {positive_train_rows}."
        )

    classifier = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=180,
                    max_depth=14,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    regressor = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=180,
                    max_depth=14,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    classifier.fit(X_train, y_binary_train)
    regressor.fit(X_train.loc[positive_train_mask], y_count_train.loc[positive_train_mask])

    event_probabilities = classifier.predict_proba(X_test)[:, 1]
    expected_count_if_event = regressor.predict(X_test)
    expected_count_if_event = [max(0.0, float(value)) for value in expected_count_if_event]

    raw_predictions = [
        max(0.0, float(probability)) * max(0.0, float(expected_count))
        for probability, expected_count in zip(event_probabilities, expected_count_if_event)
    ]

    actual_sum = float(y_count_test.sum())
    raw_predicted_sum = float(sum(raw_predictions))
    calibration_factor = actual_sum / max(raw_predicted_sum, 1.0)
    calibration_factor = max(0.001, min(float(calibration_factor), 10.0))

    predictions = [float(value) * calibration_factor for value in raw_predictions]

    predicted_binary_default = [1 if prob >= 0.5 else 0 for prob in event_probabilities]
    predicted_binary_sensitive = [1 if prob >= 0.2 else 0 for prob in event_probabilities]

    mae = float(mean_absolute_error(y_count_test, predictions))
    mse = float(mean_squared_error(y_count_test, predictions))
    rmse = math.sqrt(mse)
    r2 = float(r2_score(y_count_test, predictions)) if len(y_count_test) > 1 else 0.0

    predicted_sum = float(sum(predictions))
    aggregate_error_pct = abs(predicted_sum - actual_sum) / max(actual_sum, 1.0) * 100.0
    raw_aggregate_error_pct = abs(raw_predicted_sum - actual_sum) / max(actual_sum, 1.0) * 100.0

    try:
        roc_auc = float(roc_auc_score(y_binary_test, event_probabilities))
    except ValueError:
        roc_auc = 0.0
    try:
        average_precision = float(average_precision_score(y_binary_test, event_probabilities))
    except ValueError:
        average_precision = 0.0

    classifier_metrics = {
        "positive_rate_train": round(float(y_binary_train.mean()), 6),
        "positive_rate_test": round(float(y_binary_test.mean()), 6),
        "roc_auc": round(roc_auc, 6),
        "average_precision": round(average_precision, 6),
        "threshold_0_50": {
            "accuracy": round(float(accuracy_score(y_binary_test, predicted_binary_default)), 6),
            "precision": round(float(precision_score(y_binary_test, predicted_binary_default, zero_division=0)), 6),
            "recall": round(float(recall_score(y_binary_test, predicted_binary_default, zero_division=0)), 6),
            "f1": round(float(f1_score(y_binary_test, predicted_binary_default, zero_division=0)), 6),
        },
        "threshold_0_20": {
            "accuracy": round(float(accuracy_score(y_binary_test, predicted_binary_sensitive)), 6),
            "precision": round(float(precision_score(y_binary_test, predicted_binary_sensitive, zero_division=0)), 6),
            "recall": round(float(recall_score(y_binary_test, predicted_binary_sensitive, zero_division=0)), 6),
            "f1": round(float(f1_score(y_binary_test, predicted_binary_sensitive, zero_division=0)), 6),
        },
    }

    model_bundle = {
        "model_type": "TwoStageZeroInflatedRandomForest",
        "classifier": classifier,
        "regressor": regressor,
        "feature_columns": feature_columns,
        "numeric_features": ML_NUMERIC_FEATURES,
        "categorical_features": ML_CATEGORICAL_FEATURES,
        "target_column": ML_TARGET_COLUMN,
        "calibration_factor": calibration_factor,
        "prediction_formula": "event_probability * expected_count_if_event * calibration_factor",
    }

    generated_at = datetime.utcnow().replace(microsecond=0)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"{VOLUME_MODEL_NAME}.joblib"
    metrics_path = MODEL_DIR / f"{VOLUME_MODEL_NAME}_metrics.json"

    joblib.dump(model_bundle, model_path)

    metrics = {
        "status": "ok",
        "model_name": VOLUME_MODEL_NAME,
        "model_type": "TwoStageZeroInflatedRandomForest",
        "generated_at": generated_at.isoformat(),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "features": feature_columns,
        "target": ML_TARGET_COLUMN,
        "binary_target": "target_has_incident_next_hour",
        "row_count": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_train_rows": positive_train_rows,
        "positive_test_rows": int(y_binary_test.sum()),
        "test_size": test_size,
        "time_range": {
            "min_feature_timestamp": df["feature_timestamp"].min().isoformat(),
            "max_feature_timestamp": df["feature_timestamp"].max().isoformat(),
        },
        "metrics": {
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "r2": round(r2, 6),
            "actual_sum_test": round(actual_sum, 6),
            "raw_predicted_sum_test": round(raw_predicted_sum, 6),
            "raw_aggregate_error_pct": round(raw_aggregate_error_pct, 6),
            "predicted_sum_test": round(predicted_sum, 6),
            "aggregate_error_pct": round(aggregate_error_pct, 6),
            "calibration_factor": round(calibration_factor, 8),
        },
        "classifier_metrics": classifier_metrics,
    }

    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    storage: dict[str, Any] = {
        "local_model_path": str(model_path),
        "local_metrics_path": str(metrics_path),
    }
    if MODEL_BUCKET_NAME and MODEL_BUCKET_ENDPOINT_URL:
        storage["s3"] = upload_volume_model_artifacts(model_path=model_path, metrics_path=metrics_path)

    metrics["storage"] = storage
    return metrics


def main() -> dict[str, Any]:
    lookback_days = parse_int_env("ML_LOOKBACK_DAYS", DEFAULT_ML_LOOKBACK_DAYS, 7, 3650)
    test_size = parse_float_env("ML_TEST_SIZE", DEFAULT_ML_TEST_SIZE, 0.05, 0.4)

    started_at = datetime.utcnow().replace(microsecond=0)

    print(
        json.dumps(
            {
                "status": "running",
                "pipeline": "weekly_volume_model_training",
                "step": "fetch_dataset",
                "started_at": started_at.isoformat(),
                "lookback_days": lookback_days,
                "test_size": test_size,
            }
        ),
        flush=True,
    )

    rows = fetch_volume_ml_dataset(lookback_days=lookback_days)
    dataset_summary = summarize_volume_ml_dataset(rows, lookback_days=lookback_days)

    print(
        json.dumps(
            {
                "status": "running",
                "pipeline": "weekly_volume_model_training",
                "step": "train_model",
                "dataset": dataset_summary,
            },
            default=str,
        ),
        flush=True,
    )

    training_result = train_volume_forecast_model(rows, test_size=test_size)
    finished_at = datetime.utcnow().replace(microsecond=0)

    result = {
        "status": "ok",
        "pipeline": "weekly_volume_model_training",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "dataset": dataset_summary,
        "training": training_result,
    }

    print(json.dumps(result, indent=2, default=str), flush=True)
    return result


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error = {
            "status": "error",
            "pipeline": "weekly_volume_model_training",
            "message": str(exc),
        }
        print(json.dumps(error, indent=2, default=str), file=sys.stderr, flush=True)
        raise
