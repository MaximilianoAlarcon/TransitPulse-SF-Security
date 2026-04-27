"""
Standalone weekly training script for the CI San Francisco risk classifier model.

This script does NOT import app.py. It:
1. connects to PostgreSQL directly using DB_* env vars
2. builds a supervised ML dataset from risk_features_hourly + incident_counts_hourly
3. creates target_risk_score and target_risk_level from next-hour observed incidents
   plus severity/pressure signals
4. trains a calibrated risk classifier + score regressor
5. evaluates metrics
6. writes local artifacts
7. uploads .joblib and metrics JSON to Railway Bucket / S3-compatible storage

Usage:
    python ml_model_risk_classifier.py

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
    RISK_ML_LOOKBACK_DAYS=180
    RISK_ML_TEST_SIZE=0.2
    RISK_ML_MIN_ROWS=200
    MODEL_DIR=models
    RISK_CLASSIFIER_MODEL_NAME=risk_classifier_random_forest_v1
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
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "models"))
RISK_CLASSIFIER_MODEL_NAME = os.environ.get(
    "RISK_CLASSIFIER_MODEL_NAME",
    "risk_classifier_random_forest_v1",
)

DEFAULT_RISK_ML_LOOKBACK_DAYS = int(os.environ.get("RISK_ML_LOOKBACK_DAYS", "180"))
DEFAULT_RISK_ML_TEST_SIZE = float(os.environ.get("RISK_ML_TEST_SIZE", "0.2"))
DEFAULT_RISK_ML_MIN_ROWS = int(os.environ.get("RISK_ML_MIN_ROWS", "200"))

MODEL_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME")
MODEL_BUCKET_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL")
MODEL_BUCKET_REGION = os.environ.get("AWS_DEFAULT_REGION", "auto")
MODEL_S3_PREFIX = os.environ.get("MODEL_S3_PREFIX", "models")

RISK_NUMERIC_FEATURES = [
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

RISK_CATEGORICAL_FEATURES = [
    "police_district",
    "incident_category",
    "day_of_week",
]

RISK_SCORE_TARGET_COLUMN = "target_risk_score"
RISK_LEVEL_TARGET_COLUMN = "target_risk_level"

RISK_LEVEL_ORDER = ["Low", "Medium", "High", "Very High"]
RISK_LEVEL_TO_INT = {label: index for index, label in enumerate(RISK_LEVEL_ORDER)}
INT_TO_RISK_LEVEL = {index: label for label, index in RISK_LEVEL_TO_INT.items()}


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


def fetch_risk_classifier_dataset(lookback_days: int) -> list[dict[str, Any]]:
    """
    Build one training row per hour + district + category.

    The target is derived from the observed next hour for the same district/category:
    - next_hour_incidents
    - next_hour_open_active_count
    - next_hour_filed_online_count

    The risk target is intentionally not just volume. It combines:
    - probability/volume proxy: next hour incident happened and count size
    - category severity
    - open/active pressure
    - report-delay pressure
    - night-time context

    This gives the classifier a product-useful target until explicit human labels exist.
    """
    return fetch_all_dict(
        """
        WITH next_counts AS (
            SELECT
                bucket_start,
                police_district,
                incident_category,
                SUM(total_incidents) AS next_hour_incidents,
                SUM(open_active_count) AS next_hour_open_active_count,
                SUM(filed_online_count) AS next_hour_filed_online_count
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
            COALESCE(n.next_hour_incidents, 0) AS next_hour_incidents,
            COALESCE(n.next_hour_open_active_count, 0) AS next_hour_open_active_count,
            COALESCE(n.next_hour_filed_online_count, 0) AS next_hour_filed_online_count
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


def category_severity_score(category: Any) -> float:
    normalized = str(category or "").strip().lower()

    very_high_keywords = [
        "homicide",
        "sex offense",
        "weapons",
        "weapon",
    ]
    high_keywords = [
        "assault",
        "robbery",
        "burglary",
        "arson",
        "offences against the family",
        "offenses against the family",
    ]
    medium_keywords = [
        "motor vehicle theft",
        "larceny",
        "theft",
        "stolen property",
        "fraud",
        "malicious mischief",
        "vandalism",
        "forgery",
        "embezzlement",
    ]
    low_keywords = [
        "drug",
        "disorderly",
        "liquor",
        "gambling",
        "prostitution",
    ]

    if any(keyword in normalized for keyword in very_high_keywords):
        return 1.0
    if any(keyword in normalized for keyword in high_keywords):
        return 0.82
    if any(keyword in normalized for keyword in medium_keywords):
        return 0.58
    if any(keyword in normalized for keyword in low_keywords):
        return 0.42
    return 0.35


def is_night_hour(hour: Any) -> bool:
    try:
        h = int(hour)
    except (TypeError, ValueError):
        return False
    return h >= 22 or h <= 5


def minmax_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if math.isclose(min_value, max_value):
        return pd.Series([0.0] * len(numeric), index=numeric.index)
    return (numeric - min_value) / (max_value - min_value)


def assign_risk_level(score: float) -> str:
    if score >= 0.75:
        return "Very High"
    if score >= 0.55:
        return "High"
    if score >= 0.30:
        return "Medium"
    return "Low"


def add_risk_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["category_severity_score"] = df["incident_category"].map(category_severity_score)
    df["is_night_hour"] = df["hour_of_day"].map(is_night_hour).astype(int)

    next_hour_incidents_norm = minmax_series(df["next_hour_incidents"])
    last_24h_norm = minmax_series(df["incidents_last_24h"])
    last_7d_norm = minmax_series(df["incidents_last_7d"])
    delay_norm = minmax_series(df["avg_report_delay_minutes_24h"])

    open_pressure = pd.to_numeric(df["open_active_ratio_24h"], errors="coerce").fillna(0.0).clip(0, 1)
    filed_online_ratio = pd.to_numeric(df["filed_online_ratio_24h"], errors="coerce").fillna(0.0).clip(0, 1)
    severity = pd.to_numeric(df["category_severity_score"], errors="coerce").fillna(0.35).clip(0, 1)
    night = pd.to_numeric(df["is_night_hour"], errors="coerce").fillna(0.0).clip(0, 1)

    # Product-oriented synthetic risk score until human labels exist.
    # The target is intentionally bounded [0,1].
    df[RISK_SCORE_TARGET_COLUMN] = (
        0.34 * next_hour_incidents_norm
        + 0.19 * last_24h_norm
        + 0.11 * last_7d_norm
        + 0.16 * severity
        + 0.10 * open_pressure
        + 0.05 * delay_norm
        + 0.03 * night
        + 0.02 * (1.0 - filed_online_ratio)
    ).clip(0, 1)

    df[RISK_LEVEL_TARGET_COLUMN] = df[RISK_SCORE_TARGET_COLUMN].map(assign_risk_level)
    df["target_risk_level_id"] = df[RISK_LEVEL_TARGET_COLUMN].map(RISK_LEVEL_TO_INT).astype(int)

    return df


def summarize_risk_dataset(df: pd.DataFrame, lookback_days: int) -> dict[str, Any]:
    if df.empty:
        return {
            "lookback_days": lookback_days,
            "row_count": 0,
            "min_feature_timestamp": None,
            "max_feature_timestamp": None,
            "district_count": 0,
            "category_count": 0,
            "risk_level_counts": {},
            "target_risk_score_avg": 0,
            "target_risk_score_max": 0,
        }

    return {
        "lookback_days": lookback_days,
        "row_count": int(len(df)),
        "min_feature_timestamp": df["feature_timestamp"].min().isoformat(),
        "max_feature_timestamp": df["feature_timestamp"].max().isoformat(),
        "district_count": int(df["police_district"].nunique()),
        "category_count": int(df["incident_category"].nunique()),
        "risk_level_counts": {
            str(key): int(value)
            for key, value in df[RISK_LEVEL_TARGET_COLUMN].value_counts().sort_index().to_dict().items()
        },
        "target_risk_score_avg": round(float(df[RISK_SCORE_TARGET_COLUMN].mean()), 6),
        "target_risk_score_max": round(float(df[RISK_SCORE_TARGET_COLUMN].max()), 6),
        "next_hour_incidents_sum": round(float(df["next_hour_incidents"].sum()), 6),
        "next_hour_incidents_max": round(float(df["next_hour_incidents"].max()), 6),
    }


def get_model_artifact_keys() -> dict[str, str]:
    prefix = MODEL_S3_PREFIX.rstrip("/")
    return {
        "model": f"{prefix}/{RISK_CLASSIFIER_MODEL_NAME}.joblib",
        "metrics": f"{prefix}/{RISK_CLASSIFIER_MODEL_NAME}_metrics.json",
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


def upload_risk_model_artifacts(model_path: Path, metrics_path: Path) -> dict[str, Any]:
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
            ("categorical", encoder, RISK_CATEGORICAL_FEATURES),
            ("numeric", "passthrough", RISK_NUMERIC_FEATURES),
        ]
    )


def train_risk_classifier_model(rows: list[dict[str, Any]], test_size: float, min_rows: int) -> dict[str, Any]:
    if len(rows) < min_rows:
        raise ValueError(f"Not enough rows to train. Required at least {min_rows}, got {len(rows)}.")

    df = pd.DataFrame(rows).sort_values("feature_timestamp").reset_index(drop=True)

    for column in RISK_NUMERIC_FEATURES + [
        "next_hour_incidents",
        "next_hour_open_active_count",
        "next_hour_filed_online_count",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    for column in RISK_CATEGORICAL_FEATURES:
        df[column] = df[column].fillna("Unknown").astype(str)

    df = add_risk_targets(df)

    feature_columns = RISK_NUMERIC_FEATURES + RISK_CATEGORICAL_FEATURES
    X = df[feature_columns]
    y_level = df["target_risk_level_id"]
    y_score = df[RISK_SCORE_TARGET_COLUMN]

    split_index = int(len(df) * (1.0 - test_size))
    split_index = max(1, min(split_index, len(df) - 1))

    X_train = X.iloc[:split_index]
    y_level_train = y_level.iloc[:split_index]
    y_score_train = y_score.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_level_test = y_level.iloc[split_index:]
    y_score_test = y_score.iloc[split_index:]

    train_level_count = int(y_level_train.nunique())
    test_level_count = int(y_level_test.nunique())

    if train_level_count < 2:
        raise ValueError(
            "Not enough target risk level diversity to train classifier. "
            f"Distinct train levels: {train_level_count}."
        )

    classifier = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=220,
                    max_depth=16,
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
                    n_estimators=220,
                    max_depth=16,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    classifier.fit(X_train, y_level_train)
    regressor.fit(X_train, y_score_train)

    predicted_level_ids = classifier.predict(X_test)
    predicted_scores = regressor.predict(X_test)
    predicted_scores = [max(0.0, min(1.0, float(value))) for value in predicted_scores]

    accuracy = float(accuracy_score(y_level_test, predicted_level_ids))
    balanced_accuracy = float(balanced_accuracy_score(y_level_test, predicted_level_ids))
    macro_precision = float(precision_score(y_level_test, predicted_level_ids, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_level_test, predicted_level_ids, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_level_test, predicted_level_ids, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_level_test, predicted_level_ids, average="weighted", zero_division=0))

    mae = float(mean_absolute_error(y_score_test, predicted_scores))
    mse = float(mean_squared_error(y_score_test, predicted_scores))
    rmse = math.sqrt(mse)
    r2 = float(r2_score(y_score_test, predicted_scores)) if len(y_score_test) > 1 else 0.0

    labels_present = sorted(set(y_level_test.tolist()) | set(predicted_level_ids.tolist()))
    confusion = confusion_matrix(y_level_test, predicted_level_ids, labels=labels_present).tolist()
    report = classification_report(
        y_level_test,
        predicted_level_ids,
        labels=labels_present,
        target_names=[INT_TO_RISK_LEVEL.get(label, str(label)) for label in labels_present],
        output_dict=True,
        zero_division=0,
    )

    model_bundle = {
        "model_type": "RiskClassifierRandomForestV1",
        "classifier": classifier,
        "regressor": regressor,
        "feature_columns": feature_columns,
        "numeric_features": RISK_NUMERIC_FEATURES,
        "categorical_features": RISK_CATEGORICAL_FEATURES,
        "score_target_column": RISK_SCORE_TARGET_COLUMN,
        "level_target_column": RISK_LEVEL_TARGET_COLUMN,
        "risk_level_order": RISK_LEVEL_ORDER,
        "risk_level_to_int": RISK_LEVEL_TO_INT,
        "int_to_risk_level": INT_TO_RISK_LEVEL,
        "target_strategy": "synthetic_next_hour_observed_plus_severity_pressure",
        "score_formula": (
            "0.34*next_hour_incidents_norm + 0.19*last_24h_norm + "
            "0.11*last_7d_norm + 0.16*category_severity + "
            "0.10*open_pressure + 0.05*delay_norm + 0.03*night + "
            "0.02*(1-filed_online_ratio)"
        ),
    }

    generated_at = datetime.utcnow().replace(microsecond=0)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"{RISK_CLASSIFIER_MODEL_NAME}.joblib"
    metrics_path = MODEL_DIR / f"{RISK_CLASSIFIER_MODEL_NAME}_metrics.json"

    joblib.dump(model_bundle, model_path)

    dataset_summary = summarize_risk_dataset(df, lookback_days=0)

    metrics = {
        "status": "ok",
        "model_name": RISK_CLASSIFIER_MODEL_NAME,
        "model_type": "RiskClassifierRandomForestV1",
        "generated_at": generated_at.isoformat(),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "features": feature_columns,
        "score_target": RISK_SCORE_TARGET_COLUMN,
        "level_target": RISK_LEVEL_TARGET_COLUMN,
        "row_count": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": test_size,
        "time_range": {
            "min_feature_timestamp": df["feature_timestamp"].min().isoformat(),
            "max_feature_timestamp": df["feature_timestamp"].max().isoformat(),
        },
        "target_distribution": dataset_summary["risk_level_counts"],
        "train_level_count": train_level_count,
        "test_level_count": test_level_count,
        "classifier_metrics": {
            "accuracy": round(accuracy, 6),
            "balanced_accuracy": round(balanced_accuracy, 6),
            "macro_precision": round(macro_precision, 6),
            "macro_recall": round(macro_recall, 6),
            "macro_f1": round(macro_f1, 6),
            "weighted_f1": round(weighted_f1, 6),
            "labels_present": [INT_TO_RISK_LEVEL.get(label, str(label)) for label in labels_present],
            "confusion_matrix": confusion,
            "classification_report": report,
        },
        "score_regression_metrics": {
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "r2": round(r2, 6),
            "actual_score_avg_test": round(float(y_score_test.mean()), 6),
            "predicted_score_avg_test": round(float(sum(predicted_scores) / len(predicted_scores)), 6)
            if predicted_scores
            else 0,
        },
        "target_strategy": model_bundle["target_strategy"],
        "score_formula": model_bundle["score_formula"],
    }

    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    storage: dict[str, Any] = {
        "local_model_path": str(model_path),
        "local_metrics_path": str(metrics_path),
    }
    if MODEL_BUCKET_NAME and MODEL_BUCKET_ENDPOINT_URL:
        storage["s3"] = upload_risk_model_artifacts(model_path=model_path, metrics_path=metrics_path)

    metrics["storage"] = storage
    return metrics


def main() -> dict[str, Any]:
    lookback_days = parse_int_env(
        "RISK_ML_LOOKBACK_DAYS",
        DEFAULT_RISK_ML_LOOKBACK_DAYS,
        7,
        3650,
    )
    test_size = parse_float_env(
        "RISK_ML_TEST_SIZE",
        DEFAULT_RISK_ML_TEST_SIZE,
        0.05,
        0.4,
    )
    min_rows = parse_int_env(
        "RISK_ML_MIN_ROWS",
        DEFAULT_RISK_ML_MIN_ROWS,
        50,
        5000000,
    )

    started_at = datetime.utcnow().replace(microsecond=0)

    print(
        json.dumps(
            {
                "status": "running",
                "pipeline": "weekly_risk_classifier_training",
                "step": "fetch_dataset",
                "started_at": started_at.isoformat(),
                "lookback_days": lookback_days,
                "test_size": test_size,
                "min_rows": min_rows,
            }
        ),
        flush=True,
    )

    rows = fetch_risk_classifier_dataset(lookback_days=lookback_days)

    raw_df = pd.DataFrame(rows)
    if not raw_df.empty:
        for column in RISK_NUMERIC_FEATURES + [
            "next_hour_incidents",
            "next_hour_open_active_count",
            "next_hour_filed_online_count",
        ]:
            raw_df[column] = pd.to_numeric(raw_df[column], errors="coerce").fillna(0)
        for column in RISK_CATEGORICAL_FEATURES:
            raw_df[column] = raw_df[column].fillna("Unknown").astype(str)
        preview_df = add_risk_targets(raw_df)
        dataset_summary = summarize_risk_dataset(preview_df, lookback_days=lookback_days)
    else:
        dataset_summary = summarize_risk_dataset(pd.DataFrame(), lookback_days=lookback_days)

    print(
        json.dumps(
            {
                "status": "running",
                "pipeline": "weekly_risk_classifier_training",
                "step": "train_model",
                "dataset": dataset_summary,
            },
            default=str,
        ),
        flush=True,
    )

    training_result = train_risk_classifier_model(
        rows,
        test_size=test_size,
        min_rows=min_rows,
    )
    finished_at = datetime.utcnow().replace(microsecond=0)

    result = {
        "status": "ok",
        "pipeline": "weekly_risk_classifier_training",
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
            "pipeline": "weekly_risk_classifier_training",
            "message": str(exc),
        }
        print(json.dumps(error, indent=2, default=str), file=sys.stderr, flush=True)
        raise
