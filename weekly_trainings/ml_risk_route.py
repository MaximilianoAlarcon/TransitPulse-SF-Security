"""
Standalone weekly training script for CI San Francisco leg-level route incident probability model.

This script does not import app.py or ml_utils.py. It generates its own training
rows at the leg level from incidents_raw, trains a probability model, calculates
metrics, writes local artifacts, and uploads them to S3/Railway Bucket.

The deployed endpoint can then return probability of incidents for each leg of an
itinerary, with WALK / CAR / PUBLIC_TRANSPORT exposure handled explicitly.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "models"))
MODEL_S3_PREFIX = os.environ.get("MODEL_S3_PREFIX", "models")
ROUTE_RISK_MODEL_NAME = os.environ.get("ROUTE_RISK_MODEL_NAME", "ml_risk_route")
MODEL_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME")
MODEL_BUCKET_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL")
MODEL_BUCKET_REGION = os.environ.get("AWS_DEFAULT_REGION", "auto")

LOOKBACK_DAYS = int(os.environ.get("ROUTE_RISK_LOOKBACK_DAYS", "180"))
TARGET_ROWS = int(os.environ.get("ROUTE_LEG_RISK_TARGET_ROWS", os.environ.get("ROUTE_RISK_TARGET_ROWS", "900")))
ROUTE_POOL_SIZE = int(os.environ.get("ROUTE_RISK_ROUTE_POOL_SIZE", "900"))
TEST_SIZE = float(os.environ.get("ROUTE_RISK_TEST_SIZE", "0.2"))
MIN_ROWS = int(os.environ.get("ROUTE_RISK_MIN_ROWS", "120"))
HOURS = [int(x.strip()) for x in os.environ.get("ROUTE_RISK_HOURS", "2,8,12,15,18,22").split(",") if x.strip().isdigit()]
DAYS = [x.strip() for x in os.environ.get("ROUTE_RISK_DAYS", "Wednesday,Thursday,Friday,Saturday").split(",") if x.strip()]

NUMERIC_FEATURES = [
    "travel_hour",
    "leg_duration_sec",
    "leg_distance_m",
    "incidents_near_leg_100m_24h",
    "incidents_near_leg_250m_24h",
    "incidents_near_leg_500m_24h",
    "incidents_near_leg_7d",
    "theft_ratio_near_leg_7d",
    "assault_ratio_near_leg_7d",
    "night_ratio_near_leg_7d",
    "avg_distance_incidents_m",
    "max_segment_density",
    "mode_exposure_factor",
    "is_walk",
    "is_car",
    "is_public_transport",
    "num_transfers_before_leg",
    "leg_sequence_ratio",
]
CATEGORICAL_FEATURES = ["travel_day_of_week", "transport_mode"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "target_has_incident"


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


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


def make_preprocessor() -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("categorical", encoder, CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )


def build_leg_training_dataset(target_rows: int, lookback_days: int, route_pool_size: int) -> list[dict[str, Any]]:
    safe_hours = [h for h in HOURS if 0 <= h <= 23] or [2, 8, 12, 15, 18, 22]
    safe_days = DAYS or ["Wednesday", "Thursday", "Friday", "Saturday"]
    target_rows = max(int(target_rows), MIN_ROWS)
    route_pool_size = max(int(route_pool_size), target_rows * 2, 200)

    # Synthetic leg geometries are made from pairs of real incident locations.
    # That gives a useful spatial mix while keeping the script fully standalone.
    rows = fetch_all_dict(
        """
        WITH source_points AS (
            SELECT i.geom, ROW_NUMBER() OVER (ORDER BY random()) AS rn
            FROM incidents_raw i
            WHERE i.geom IS NOT NULL
              AND i.incident_datetime >= NOW() - (%s::int * INTERVAL '1 day')
              AND ST_X(i.geom) BETWEEN -123.0 AND -121.8
              AND ST_Y(i.geom) BETWEEN 37.0 AND 38.3
            ORDER BY random()
            LIMIT %s
        ), paired_points AS (
            SELECT
                a.geom AS origin_geom,
                b.geom AS dest_geom,
                ST_MakeLine(a.geom, b.geom) AS leg_geom,
                ST_Distance(a.geom::geography, b.geom::geography) AS leg_distance_m
            FROM source_points a
            JOIN source_points b
              ON b.rn = ((a.rn + (%s::int / 3)) %% %s::int) + 1
            WHERE a.rn <> b.rn
              AND ST_Distance(a.geom::geography, b.geom::geography) BETWEEN 80 AND 16000
        ), hours AS (
            SELECT UNNEST(%s::int[]) AS travel_hour
        ), days AS (
            SELECT UNNEST(%s::text[]) AS travel_day_of_week
        ), modes AS (
            SELECT * FROM (VALUES ('WALK'::text), ('CAR'::text), ('PUBLIC_TRANSPORT'::text)) AS m(transport_mode)
        ), candidates AS (
            SELECT
                p.leg_geom,
                p.leg_distance_m,
                h.travel_hour,
                d.travel_day_of_week,
                m.transport_mode,
                ROW_NUMBER() OVER (ORDER BY random()) AS sample_rank
            FROM paired_points p
            CROSS JOIN hours h
            CROSS JOIN days d
            CROSS JOIN modes m
        ), selected AS (
            SELECT *
            FROM candidates
            WHERE sample_rank <= %s
        ), features AS (
            SELECT
                travel_hour,
                travel_day_of_week,
                transport_mode,
                leg_distance_m,
                CASE
                    WHEN transport_mode = 'WALK' THEN leg_distance_m / 1.35
                    WHEN transport_mode = 'CAR' THEN leg_distance_m / 8.0
                    ELSE leg_distance_m / 5.0
                END AS leg_duration_sec,
                CASE WHEN transport_mode = 'WALK' THEN 1.0 ELSE 0.0 END AS is_walk,
                CASE WHEN transport_mode = 'CAR' THEN 1.0 ELSE 0.0 END AS is_car,
                CASE WHEN transport_mode = 'PUBLIC_TRANSPORT' THEN 1.0 ELSE 0.0 END AS is_public_transport,
                CASE
                    WHEN transport_mode = 'WALK' THEN 1.00
                    WHEN transport_mode = 'PUBLIC_TRANSPORT' THEN 0.45
                    WHEN transport_mode = 'CAR' THEN 0.18
                    ELSE 0.50
                END AS mode_exposure_factor,
                CASE WHEN transport_mode = 'PUBLIC_TRANSPORT' THEN floor(random() * 3)::int ELSE 0 END AS num_transfers_before_leg,
                random() AS leg_sequence_ratio,
                f.*
            FROM selected s
            CROSS JOIN LATERAL (
                WITH leg AS (SELECT s.leg_geom AS geom),
                incidents_24h AS (
                    SELECT i.*
                    FROM incidents_raw i, leg l
                    WHERE i.geom IS NOT NULL
                      AND i.incident_datetime >= NOW() - INTERVAL '24 hours'
                      AND ST_DWithin(i.geom::geography, l.geom::geography, 500)
                ), incidents_7d AS (
                    SELECT i.*
                    FROM incidents_raw i, leg l
                    WHERE i.geom IS NOT NULL
                      AND i.incident_datetime >= NOW() - INTERVAL '7 days'
                      AND ST_DWithin(i.geom::geography, l.geom::geography, 500)
                ), segments AS (
                    SELECT ST_LineSubstring(l.geom, gs::float8, LEAST((gs + 0.10)::float8, 1.0)) AS geom
                    FROM leg l, generate_series(0.0, 0.90, 0.10) AS gs
                ), segment_counts AS (
                    SELECT COUNT(i.row_id) AS incident_count
                    FROM segments seg
                    LEFT JOIN incidents_raw i
                      ON i.geom IS NOT NULL
                     AND i.incident_datetime >= NOW() - INTERVAL '24 hours'
                     AND ST_DWithin(i.geom::geography, seg.geom::geography, 250)
                    GROUP BY seg.geom
                )
                SELECT
                    (SELECT COUNT(*) FROM incidents_24h i, leg l WHERE ST_DWithin(i.geom::geography, l.geom::geography, 100))::int AS incidents_near_leg_100m_24h,
                    (SELECT COUNT(*) FROM incidents_24h i, leg l WHERE ST_DWithin(i.geom::geography, l.geom::geography, 250))::int AS incidents_near_leg_250m_24h,
                    (SELECT COUNT(*) FROM incidents_24h)::int AS incidents_near_leg_500m_24h,
                    (SELECT COUNT(*) FROM incidents_7d)::int AS incidents_near_leg_7d,
                    COALESCE((SELECT AVG(CASE WHEN incident_category ILIKE '%%theft%%' OR incident_category ILIKE '%%larceny%%' THEN 1.0 ELSE 0.0 END) FROM incidents_7d), 0.0) AS theft_ratio_near_leg_7d,
                    COALESCE((SELECT AVG(CASE WHEN incident_category ILIKE '%%assault%%' THEN 1.0 ELSE 0.0 END) FROM incidents_7d), 0.0) AS assault_ratio_near_leg_7d,
                    COALESCE((SELECT AVG(CASE WHEN EXTRACT(HOUR FROM incident_datetime) >= 20 OR EXTRACT(HOUR FROM incident_datetime) <= 5 THEN 1.0 ELSE 0.0 END) FROM incidents_7d), 0.0) AS night_ratio_near_leg_7d,
                    COALESCE((SELECT AVG(ST_Distance(i.geom::geography, l.geom::geography)) FROM incidents_7d i, leg l), 9999.0) AS avg_distance_incidents_m,
                    COALESCE((SELECT MAX(incident_count)::double precision FROM segment_counts), 0.0) AS max_segment_density
            ) f
        ), labeled AS (
            SELECT *,
                LEAST(1.0, GREATEST(0.0,
                    (
                        0.27 * LEAST(incidents_near_leg_250m_24h::double precision / 8.0, 1.0)
                      + 0.17 * LEAST(incidents_near_leg_7d::double precision / 90.0, 1.0)
                      + 0.16 * LEAST(max_segment_density::double precision / 6.0, 1.0)
                      + 0.10 * (1.0 - LEAST(avg_distance_incidents_m / 500.0, 1.0))
                      + 0.08 * LEAST(night_ratio_near_leg_7d, 1.0)
                      + 0.06 * LEAST(theft_ratio_near_leg_7d, 1.0)
                      + 0.07 * LEAST(assault_ratio_near_leg_7d, 1.0)
                      + CASE WHEN travel_hour >= 22 OR travel_hour <= 5 THEN 0.08 WHEN travel_hour BETWEEN 18 AND 21 THEN 0.05 ELSE 0.02 END
                    )
                    * GREATEST(0.25, LEAST(1.50, 0.60 + 0.75 * mode_exposure_factor + 0.08 * LEAST(num_transfers_before_leg::double precision / 3.0, 1.0)))
                )) AS target_incident_probability
            FROM features
        )
        SELECT
            travel_hour,
            travel_day_of_week,
            transport_mode,
            leg_duration_sec,
            leg_distance_m,
            incidents_near_leg_100m_24h,
            incidents_near_leg_250m_24h,
            incidents_near_leg_500m_24h,
            incidents_near_leg_7d,
            theft_ratio_near_leg_7d,
            assault_ratio_near_leg_7d,
            night_ratio_near_leg_7d,
            avg_distance_incidents_m,
            max_segment_density,
            mode_exposure_factor,
            is_walk,
            is_car,
            is_public_transport,
            num_transfers_before_leg,
            leg_sequence_ratio,
            target_incident_probability,
            CASE WHEN random() < target_incident_probability THEN 1 ELSE 0 END AS target_has_incident
        FROM labeled;
        """,
        (lookback_days, route_pool_size, route_pool_size, route_pool_size, safe_hours, safe_days, target_rows),
    )
    return rows


def train_model(rows: list[dict[str, Any]], test_size: float, min_rows: int) -> dict[str, Any]:
    if len(rows) < min_rows:
        raise ValueError(f"Not enough leg-level route risk rows to train. Required at least {min_rows}, got {len(rows)}.")

    df = pd.DataFrame(rows).reset_index(drop=True)
    for column in NUMERIC_FEATURES + ["target_incident_probability", TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].fillna("UNKNOWN").astype(str)

    y = df[TARGET_COLUMN].astype(int)
    if y.nunique() < 2:
        # Deterministic fallback: threshold the generated probability if random draw was unlucky.
        y = (df["target_incident_probability"] >= float(df["target_incident_probability"].median())).astype(int)
    if y.nunique() < 2:
        raise ValueError("Generated target still has a single class; increase ROUTE_LEG_RISK_TARGET_ROWS or route diversity.")

    split_index = int(len(df) * (1.0 - test_size))
    split_index = max(1, min(split_index, len(df) - 1))
    X = df[FEATURE_COLUMNS]
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    pipeline = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=280,
            max_depth=14,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    metrics = {
        "status": "ok",
        "model_name": ROUTE_RISK_MODEL_NAME,
        "model_type": "RouteLegIncidentProbabilityClassifier",
        "trained_at": utc_now_iso(),
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_rate": round(float(y.mean()), 6),
        "test_positive_rate": round(float(y_test.mean()), 6),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 6),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 6),
        "brier_score": round(float(brier_score_loss(y_test, probabilities)), 6),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 6) if y_test.nunique() > 1 else None,
        "average_precision": round(float(average_precision_score(y_test, probabilities)), 6) if y_test.nunique() > 1 else None,
        "predicted_probability_avg": round(float(probabilities.mean()), 6),
        "features": FEATURE_COLUMNS,
        "dataset": {
            "lookback_days": LOOKBACK_DAYS,
            "row_count": int(len(df)),
            "transport_mode_counts": {str(k): int(v) for k, v in df["transport_mode"].value_counts().to_dict().items()},
            "travel_hours": sorted(int(v) for v in df["travel_hour"].dropna().unique().tolist()),
            "target_probability_min": round(float(df["target_incident_probability"].min()), 6),
            "target_probability_avg": round(float(df["target_incident_probability"].mean()), 6),
            "target_probability_max": round(float(df["target_incident_probability"].max()), 6),
        },
    }

    bundle = {
        "model_type": metrics["model_type"],
        "model_name": ROUTE_RISK_MODEL_NAME,
        "trained_at": metrics["trained_at"],
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET_COLUMN,
        "probability_output": "leg_incident_probability",
        "pipeline": pipeline,
        "risk_level_thresholds": {"medium": 0.30, "high": 0.55, "very_high": 0.75},
    }
    return {"bundle": bundle, "metrics": metrics}


def artifact_keys() -> dict[str, str]:
    prefix = MODEL_S3_PREFIX.rstrip("/")
    return {
        "model": f"{prefix}/{ROUTE_RISK_MODEL_NAME}.joblib",
        "metrics": f"{prefix}/{ROUTE_RISK_MODEL_NAME}_metrics.json",
    }


def upload_bytes_to_bucket(key: str, data: bytes, content_type: str) -> None:
    import boto3
    if not MODEL_BUCKET_NAME:
        raise RuntimeError("AWS_S3_BUCKET_NAME is not configured.")
    if not MODEL_BUCKET_ENDPOINT_URL:
        raise RuntimeError("AWS_ENDPOINT_URL is not configured.")
    client = boto3.client(
        "s3",
        endpoint_url=MODEL_BUCKET_ENDPOINT_URL,
        region_name=MODEL_BUCKET_REGION,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
    client.put_object(Bucket=MODEL_BUCKET_NAME, Key=key, Body=data, ContentType=content_type)


def save_and_upload(bundle: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{ROUTE_RISK_MODEL_NAME}.joblib"
    metrics_path = MODEL_DIR / f"{ROUTE_RISK_MODEL_NAME}_metrics.json"
    joblib.dump(bundle, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    keys = artifact_keys()
    upload_bytes_to_bucket(keys["model"], model_path.read_bytes(), "application/octet-stream")
    upload_bytes_to_bucket(keys["metrics"], metrics_path.read_bytes(), "application/json")
    return {
        "local_artifacts": {"model_path": str(model_path), "metrics_path": str(metrics_path)},
        "s3_artifacts": {
            "bucket": MODEL_BUCKET_NAME,
            "endpoint_url": MODEL_BUCKET_ENDPOINT_URL,
            "model_key": keys["model"],
            "metrics_key": keys["metrics"],
        },
    }


def main() -> None:
    try:
        rows = build_leg_training_dataset(TARGET_ROWS, LOOKBACK_DAYS, ROUTE_POOL_SIZE)
        result = train_model(rows, TEST_SIZE, MIN_ROWS)
        artifacts = save_and_upload(result["bundle"], result["metrics"])
        output = {**result["metrics"], **artifacts}
        print(json.dumps(output, indent=2, default=str))
    except Exception as exc:
        error = {
            "status": "error",
            "model_name": ROUTE_RISK_MODEL_NAME,
            "message": str(exc),
            "failed_at": utc_now_iso(),
        }
        print(json.dumps(error, indent=2, default=str))
        raise


if __name__ == "__main__":
    main()
