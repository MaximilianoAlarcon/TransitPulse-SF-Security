"""
Standalone weekly training script for the CI San Francisco route-risk model.

This script does NOT import app.py or project utilities. It:
1. connects to PostgreSQL directly using DB_* env vars
2. builds the ML dataset from route_risk_features
3. trains ml_risk_route
4. writes local artifacts
5. uploads .joblib and metrics JSON to Railway Bucket / S3-compatible storage

Usage:
    python ml_risk_route.py

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
    MODEL_DIR=models
    ROUTE_RISK_MODEL_NAME=ml_risk_route
    MODEL_S3_PREFIX=models
    ROUTE_RISK_LOOKBACK_DAYS=180
    ROUTE_RISK_TEST_SIZE=0.2
    ROUTE_RISK_MIN_ROWS=50
    ROUTE_RISK_ALLOW_WEAK_LABELS=true
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "models"))
ROUTE_RISK_MODEL_NAME = os.environ.get("ROUTE_RISK_MODEL_NAME", "ml_risk_route")
MODEL_S3_PREFIX = os.environ.get("MODEL_S3_PREFIX", "models")
MODEL_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME")
MODEL_BUCKET_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL")
MODEL_BUCKET_REGION = os.environ.get("AWS_DEFAULT_REGION", "auto")

DEFAULT_LOOKBACK_DAYS = int(os.environ.get("ROUTE_RISK_LOOKBACK_DAYS", "180"))
DEFAULT_TEST_SIZE = float(os.environ.get("ROUTE_RISK_TEST_SIZE", "0.2"))
DEFAULT_MIN_ROWS = int(os.environ.get("ROUTE_RISK_MIN_ROWS", "50"))
ALLOW_WEAK_LABELS = os.environ.get("ROUTE_RISK_ALLOW_WEAK_LABELS", "true").strip().lower() in {"1", "true", "yes", "y"}

ROUTE_NUMERIC_FEATURES = [
    "travel_hour",
    "incidents_near_route_100m_24h",
    "incidents_near_route_250m_24h",
    "incidents_near_route_500m_24h",
    "incidents_near_route_7d",
    "theft_ratio_near_route_7d",
    "assault_ratio_near_route_7d",
    "night_ratio_near_route_7d",
    "avg_distance_incidents_m",
    "max_segment_density",
]

ROUTE_CATEGORICAL_FEATURES = ["travel_day_of_week"]
TARGET_COLUMN = "target_risk_score"


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


def parse_float_env(name: str, default: float, min_value: float, max_value: float) -> float:
    raw = os.environ.get(name)
    try:
        value = float(raw) if raw is not None else default
    except ValueError:
        value = default
    return max(min_value, min(value, max_value))


def parse_int_env(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.environ.get(name)
    try:
        value = int(raw) if raw is not None else default
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


def fetch_route_risk_dataset(lookback_days: int) -> list[dict[str, Any]]:
    return fetch_all_dict(
        """
        SELECT
            rrf.route_feature_id,
            rrf.route_id,
            rrf.computed_at,
            COALESCE(rrf.travel_hour, 12) AS travel_hour,
            COALESCE(rrf.travel_day_of_week, 'Unknown') AS travel_day_of_week,
            COALESCE(rrf.incidents_near_route_100m_24h, 0) AS incidents_near_route_100m_24h,
            COALESCE(rrf.incidents_near_route_250m_24h, 0) AS incidents_near_route_250m_24h,
            COALESCE(rrf.incidents_near_route_500m_24h, 0) AS incidents_near_route_500m_24h,
            COALESCE(rrf.incidents_near_route_7d, 0) AS incidents_near_route_7d,
            COALESCE(rrf.theft_ratio_near_route_7d, 0) AS theft_ratio_near_route_7d,
            COALESCE(rrf.assault_ratio_near_route_7d, 0) AS assault_ratio_near_route_7d,
            COALESCE(rrf.night_ratio_near_route_7d, 0) AS night_ratio_near_route_7d,
            COALESCE(rrf.avg_distance_incidents_m, 9999) AS avg_distance_incidents_m,
            COALESCE(rrf.max_segment_density, 0) AS max_segment_density,
            rrf.target_risk_score,
            rrf.target_risk_level
        FROM route_risk_features rrf
        WHERE rrf.computed_at >= NOW() - (%s::int * INTERVAL '1 day')
        ORDER BY rrf.computed_at ASC, rrf.route_feature_id ASC;
        """,
        (lookback_days,),
    )


def minmax(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
    min_value = float(clean.min())
    max_value = float(clean.max())
    if math.isclose(max_value, min_value):
        return pd.Series([0.0] * len(clean), index=clean.index)
    return (clean - min_value) / (max_value - min_value)


def build_weak_target_scores(df: pd.DataFrame) -> pd.Series:
    """
    Fallback weak labels when target_risk_score is not populated yet.

    This lets the weekly training architecture exist from day one. As soon as
    target_risk_score is populated with stronger labels, the script uses those
    real labels automatically.
    """
    density_250 = minmax(df["incidents_near_route_250m_24h"])
    density_7d = minmax(df["incidents_near_route_7d"])
    max_segment = minmax(df["max_segment_density"])
    avg_distance = pd.to_numeric(df["avg_distance_incidents_m"], errors="coerce").fillna(9999).astype(float)
    distance_pressure = 1.0 - minmax(avg_distance)
    night = pd.to_numeric(df["night_ratio_near_route_7d"], errors="coerce").fillna(0).clip(0, 1)
    theft = pd.to_numeric(df["theft_ratio_near_route_7d"], errors="coerce").fillna(0).clip(0, 1)
    assault = pd.to_numeric(df["assault_ratio_near_route_7d"], errors="coerce").fillna(0).clip(0, 1)

    score = (
        0.25 * density_250
        + 0.22 * density_7d
        + 0.20 * max_segment
        + 0.13 * distance_pressure
        + 0.08 * night
        + 0.06 * theft
        + 0.06 * assault
    )
    return score.clip(0.0, 1.0)


def level_from_score(score: float) -> str:
    value = max(0.0, min(1.0, float(score or 0)))
    if value >= 0.75:
        return "Very High"
    if value >= 0.55:
        return "High"
    if value >= 0.30:
        return "Medium"
    return "Low"


def make_preprocessor() -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("categorical", encoder, ROUTE_CATEGORICAL_FEATURES),
            ("numeric", "passthrough", ROUTE_NUMERIC_FEATURES),
        ]
    )


def summarize_dataset(rows: list[dict[str, Any]], lookback_days: int, label_source: str | None = None) -> dict[str, Any]:
    if not rows:
        return {"lookback_days": lookback_days, "row_count": 0, "label_source": label_source}
    timestamps = [row["computed_at"] for row in rows if row.get("computed_at")]
    targets = [row.get(TARGET_COLUMN) for row in rows if row.get(TARGET_COLUMN) is not None]
    return {
        "lookback_days": lookback_days,
        "row_count": len(rows),
        "min_computed_at": min(timestamps).isoformat() if timestamps else None,
        "max_computed_at": max(timestamps).isoformat() if timestamps else None,
        "explicit_target_rows": len(targets),
        "label_source": label_source,
    }


def train_route_risk_model(rows: list[dict[str, Any]], test_size: float, min_rows: int) -> dict[str, Any]:
    if len(rows) < min_rows:
        raise ValueError(f"Not enough route_risk_features rows to train. Required at least {min_rows}, got {len(rows)}.")

    df = pd.DataFrame(rows).sort_values("computed_at").reset_index(drop=True)

    for column in ROUTE_NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    df["avg_distance_incidents_m"] = pd.to_numeric(df["avg_distance_incidents_m"], errors="coerce").fillna(9999)

    for column in ROUTE_CATEGORICAL_FEATURES:
        df[column] = df[column].fillna("Unknown").astype(str)

    explicit_target_mask = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").notna()
    explicit_target_count = int(explicit_target_mask.sum())

    if explicit_target_count >= max(20, int(len(df) * 0.25)):
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0).clip(0, 1)
        label_source = "target_risk_score"
    elif ALLOW_WEAK_LABELS:
        df[TARGET_COLUMN] = build_weak_target_scores(df)
        label_source = "weak_labels_from_route_features"
    else:
        raise ValueError(
            "target_risk_score is not populated enough and ROUTE_RISK_ALLOW_WEAK_LABELS=false. "
            f"Explicit target rows: {explicit_target_count}."
        )

    feature_columns = ROUTE_NUMERIC_FEATURES + ROUTE_CATEGORICAL_FEATURES
    X = df[feature_columns]
    y = df[TARGET_COLUMN].astype(float).clip(0, 1)

    split_index = int(len(df) * (1.0 - test_size))
    split_index = max(1, min(split_index, len(df) - 1))

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=220,
                    max_depth=12,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = [max(0.0, min(1.0, float(value))) for value in pipeline.predict(X_test)]

    mae = float(mean_absolute_error(y_test, predictions))
    mse = float(mean_squared_error(y_test, predictions))
    rmse = math.sqrt(mse)
    r2 = float(r2_score(y_test, predictions)) if len(y_test) > 1 else 0.0

    bundle = {
        "model_type": "RouteRiskRandomForestRegressor",
        "model_name": ROUTE_RISK_MODEL_NAME,
        "trained_at": utc_now_iso(),
        "label_source": label_source,
        "feature_columns": feature_columns,
        "numeric_features": ROUTE_NUMERIC_FEATURES,
        "categorical_features": ROUTE_CATEGORICAL_FEATURES,
        "pipeline": pipeline,
        "risk_level_thresholds": {
            "medium": 0.30,
            "high": 0.55,
            "very_high": 0.75,
        },
    }

    metrics = {
        "status": "ok",
        "model_name": ROUTE_RISK_MODEL_NAME,
        "model_type": bundle["model_type"],
        "trained_at": bundle["trained_at"],
        "label_source": label_source,
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "explicit_target_rows": explicit_target_count,
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 6),
        "test_actual_avg": round(float(y_test.mean()), 6),
        "test_predicted_avg": round(float(sum(predictions) / max(len(predictions), 1)), 6),
        "predicted_level_counts": {},
        "features": feature_columns,
    }

    for predicted in predictions:
        level = level_from_score(predicted)
        metrics["predicted_level_counts"][level] = metrics["predicted_level_counts"].get(level, 0) + 1

    return {"bundle": bundle, "metrics": metrics, "label_source": label_source}


def get_artifact_keys() -> dict[str, str]:
    prefix = MODEL_S3_PREFIX.rstrip("/")
    return {
        "model": f"{prefix}/{ROUTE_RISK_MODEL_NAME}.joblib",
        "metrics": f"{prefix}/{ROUTE_RISK_MODEL_NAME}_metrics.json",
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
    client.put_object(Bucket=MODEL_BUCKET_NAME, Key=key, Body=data, ContentType=content_type)


def upload_artifacts(model_path: Path, metrics_path: Path) -> dict[str, Any]:
    keys = get_artifact_keys()
    upload_bytes_to_model_bucket(keys["model"], model_path.read_bytes(), "application/octet-stream")
    upload_bytes_to_model_bucket(keys["metrics"], metrics_path.read_bytes(), "application/json")
    return {
        "bucket": MODEL_BUCKET_NAME,
        "endpoint_url": MODEL_BUCKET_ENDPOINT_URL,
        "model_key": keys["model"],
        "metrics_key": keys["metrics"],
    }


def main() -> dict[str, Any]:
    lookback_days = parse_int_env("ROUTE_RISK_LOOKBACK_DAYS", DEFAULT_LOOKBACK_DAYS, 7, 3650)
    test_size = parse_float_env("ROUTE_RISK_TEST_SIZE", DEFAULT_TEST_SIZE, 0.05, 0.4)
    min_rows = parse_int_env("ROUTE_RISK_MIN_ROWS", DEFAULT_MIN_ROWS, 10, 100000)

    rows = fetch_route_risk_dataset(lookback_days)
    training_result = train_route_risk_model(rows, test_size=test_size, min_rows=min_rows)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{ROUTE_RISK_MODEL_NAME}.joblib"
    metrics_path = MODEL_DIR / f"{ROUTE_RISK_MODEL_NAME}_metrics.json"

    joblib.dump(training_result["bundle"], model_path)

    metrics = dict(training_result["metrics"])
    metrics["dataset"] = summarize_dataset(rows, lookback_days, training_result["label_source"])
    metrics["local_artifacts"] = {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
    }

    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    upload_info = upload_artifacts(model_path, metrics_path)
    metrics["s3_artifacts"] = upload_info
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    print(json.dumps(metrics, indent=2, default=str), flush=True)
    return metrics


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "model_name": ROUTE_RISK_MODEL_NAME,
                    "message": str(exc),
                    "failed_at": utc_now_iso(),
                },
                indent=2,
                default=str,
            ),
            flush=True,
        )
        raise
