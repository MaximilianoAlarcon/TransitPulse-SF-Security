"""
Standalone weekly training script for CI San Francisco route-risk model.

This script does not import app.py or ml_utils.py. It can bootstrap its own
route-risk training dataset from incidents_raw, then train and upload the
ml_risk_route model to S3/Railway Bucket.

Usage:
    python ml_risk_route.py

Required DB env vars:
    DB_HOST DB_NAME DB_USER DB_PASSWORD DB_PORT

Required for artifact upload:
    AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_DEFAULT_REGION AWS_ENDPOINT_URL AWS_S3_BUCKET_NAME

Optional env vars:
    MODEL_DIR=models
    MODEL_S3_PREFIX=models
    ROUTE_RISK_MODEL_NAME=ml_risk_route
    ROUTE_RISK_LOOKBACK_DAYS=180
    ROUTE_RISK_TEST_SIZE=0.2
    ROUTE_RISK_MIN_ROWS=50
    ROUTE_RISK_AUTO_BUILD_DATASET=true
    ROUTE_RISK_TARGET_ROWS=180
    ROUTE_RISK_HOURS=2,8,12,15,18,22
    ROUTE_RISK_DAYS=Wednesday,Thursday,Friday,Saturday
    ROUTE_RISK_ROUTE_POOL_SIZE=600
    ROUTE_RISK_PURGE_SYNTHETIC=false
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
MODEL_S3_PREFIX = os.environ.get("MODEL_S3_PREFIX", "models")
ROUTE_RISK_MODEL_NAME = os.environ.get("ROUTE_RISK_MODEL_NAME", "ml_risk_route")
MODEL_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME")
MODEL_BUCKET_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL")
MODEL_BUCKET_REGION = os.environ.get("AWS_DEFAULT_REGION", "auto")

DEFAULT_LOOKBACK_DAYS = int(os.environ.get("ROUTE_RISK_LOOKBACK_DAYS", "180"))
DEFAULT_TEST_SIZE = float(os.environ.get("ROUTE_RISK_TEST_SIZE", "0.2"))
DEFAULT_MIN_ROWS = int(os.environ.get("ROUTE_RISK_MIN_ROWS", "50"))
AUTO_BUILD_DATASET = os.environ.get("ROUTE_RISK_AUTO_BUILD_DATASET", "true").strip().lower() in {"1", "true", "yes", "y"}
PURGE_SYNTHETIC = os.environ.get("ROUTE_RISK_PURGE_SYNTHETIC", "false").strip().lower() in {"1", "true", "yes", "y"}
DEFAULT_TARGET_ROWS = int(os.environ.get("ROUTE_RISK_TARGET_ROWS", "180"))
DEFAULT_ROUTE_POOL_SIZE = int(os.environ.get("ROUTE_RISK_ROUTE_POOL_SIZE", "600"))
SYNTHETIC_SOURCE = "ml_risk_route_auto_dataset"

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
    "walk_duration_sec",
    "car_duration_sec",
    "public_transport_duration_sec",
    "walk_ratio",
    "car_ratio",
    "public_transport_ratio",
    "walk_distance",
    "num_transfers",
    "mode_exposure_factor",
]

ROUTE_CATEGORICAL_FEATURES = ["travel_day_of_week", "dominant_transport_mode"]
TARGET_COLUMN = "target_risk_score"


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


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


def parse_csv_int_env(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name)
    values: list[int] = []
    if raw:
        for part in raw.split(","):
            try:
                value = int(part.strip())
            except ValueError:
                continue
            if 0 <= value <= 23 and value not in values:
                values.append(value)
    return values or list(default)


def parse_csv_text_env(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    values = [part.strip() for part in raw.split(",")] if raw else []
    values = [value for value in values if value]
    return values or list(default)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b if b > 0 else 0


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


def execute_sql(query: str, params: tuple[Any, ...] = ()) -> int:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            affected = cur.rowcount if cur.rowcount is not None else 0
        conn.commit()
        return int(affected)


def fetch_one_value(query: str, params: tuple[Any, ...] = ()) -> Any:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone()
            return row[0] if row else None


def ensure_route_risk_mode_columns() -> None:
    execute_sql(
        """
        ALTER TABLE route_risk_features
            ADD COLUMN IF NOT EXISTS walk_duration_sec double precision DEFAULT 0,
            ADD COLUMN IF NOT EXISTS car_duration_sec double precision DEFAULT 0,
            ADD COLUMN IF NOT EXISTS public_transport_duration_sec double precision DEFAULT 0,
            ADD COLUMN IF NOT EXISTS walk_ratio double precision DEFAULT 0,
            ADD COLUMN IF NOT EXISTS car_ratio double precision DEFAULT 0,
            ADD COLUMN IF NOT EXISTS public_transport_ratio double precision DEFAULT 0,
            ADD COLUMN IF NOT EXISTS walk_distance double precision DEFAULT 0,
            ADD COLUMN IF NOT EXISTS num_transfers integer DEFAULT 0,
            ADD COLUMN IF NOT EXISTS mode_exposure_factor double precision DEFAULT 0,
            ADD COLUMN IF NOT EXISTS dominant_transport_mode text DEFAULT 'UNKNOWN';
        """
    )


def count_training_rows(lookback_days: int) -> int:
    return int(fetch_one_value(
        """
        SELECT COUNT(*)
        FROM route_risk_features
        WHERE computed_at >= NOW() - (%s::int * INTERVAL '1 day')
          AND target_risk_score IS NOT NULL;
        """,
        (lookback_days,),
    ) or 0)


def purge_synthetic_dataset() -> int:
    return execute_sql(
        """
        DELETE FROM route_requests
        WHERE travel_day_of_week LIKE %s;
        """,
        (f"%|source={SYNTHETIC_SOURCE}%",),
    )


def generate_route_risk_dataset_from_incidents(
    target_rows: int,
    lookback_days: int,
    hours: list[int],
    days: list[str],
    route_pool_size: int,
) -> dict[str, Any]:
    if target_rows <= 0:
        return {"status": "skipped", "inserted_rows": 0, "reason": "target_rows <= 0"}

    safe_hours = [int(h) for h in hours if 0 <= int(h) <= 23] or [2, 8, 12, 15, 18, 22]
    safe_days = [str(day).strip() for day in days if str(day).strip()] or ["Wednesday", "Thursday", "Friday", "Saturday"]
    buckets = max(1, len(safe_hours) * len(safe_days))
    rows_per_bucket = max(1, ceil_div(target_rows, buckets))
    route_pool_size = max(route_pool_size, target_rows * 4, 100)
    before_count = count_training_rows(lookback_days)

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
                ST_MakeLine(a.geom, b.geom) AS route_geom
            FROM source_points a
            JOIN source_points b
              ON b.rn = ((a.rn + (%s::int / 3)) %% %s::int) + 1
            WHERE a.rn <> b.rn
              AND ST_Distance(a.geom::geography, b.geom::geography) BETWEEN 700 AND 18000
        ), hours AS (
            SELECT UNNEST(%s::int[]) AS travel_hour
        ), days AS (
            SELECT UNNEST(%s::text[]) AS base_day
        ), transport_modes AS (
            SELECT *
            FROM (VALUES
                ('WALK'::text),
                ('CAR'::text),
                ('PUBLIC_TRANSPORT'::text)
            ) AS m(dominant_transport_mode)
        ), route_candidates AS (
            SELECT
                p.route_geom,
                h.travel_hour,
                d.base_day,
                m.dominant_transport_mode,
                ROW_NUMBER() OVER (PARTITION BY h.travel_hour, d.base_day, m.dominant_transport_mode ORDER BY random()) AS route_rank
            FROM paired_points p
            CROSS JOIN hours h
            CROSS JOIN days d
            CROSS JOIN transport_modes m
        ), selected_routes AS (
            SELECT
                route_geom,
                travel_hour,
                dominant_transport_mode,
                CONCAT(base_day, '|source=', %s::text, '|mode=', dominant_transport_mode) AS travel_day_of_week
            FROM route_candidates
            WHERE route_rank <= GREATEST(1, CEIL(%s::numeric / 3.0)::int)
        ), inserted_routes AS (
            INSERT INTO route_requests (
                requested_at, origin_lat, origin_lon, dest_lat, dest_lon,
                route_geom, travel_hour, travel_day_of_week
            )
            SELECT
                NOW(),
                ST_Y(ST_StartPoint(route_geom)), ST_X(ST_StartPoint(route_geom)),
                ST_Y(ST_EndPoint(route_geom)), ST_X(ST_EndPoint(route_geom)),
                route_geom, travel_hour, travel_day_of_week
            FROM selected_routes
            RETURNING route_id, route_geom, travel_hour, travel_day_of_week
        ), features AS (
            SELECT
                ir.route_id,
                ir.travel_hour,
                ir.travel_day_of_week,
                f.*,
                CASE
                    WHEN ir.travel_hour >= 22 OR ir.travel_hour <= 5 THEN 1.00
                    WHEN ir.travel_hour BETWEEN 18 AND 21 THEN 0.70
                    WHEN ir.travel_hour BETWEEN 6 AND 9 THEN 0.45
                    WHEN ir.travel_hour BETWEEN 10 AND 16 THEN 0.25
                    ELSE 0.40
                END AS travel_hour_pressure,
                CASE WHEN split_part(ir.travel_day_of_week, '|', 1) IN ('Friday', 'Saturday') THEN 0.10 ELSE 0.00 END AS weekend_pressure,
                split_part(ir.travel_day_of_week, '|mode=', 2) AS dominant_transport_mode,
                ST_Length(ir.route_geom::geography) AS route_length_m,
                CASE
                    WHEN split_part(ir.travel_day_of_week, '|mode=', 2) = 'WALK' THEN ST_Length(ir.route_geom::geography) / 1.35
                    WHEN split_part(ir.travel_day_of_week, '|mode=', 2) = 'CAR' THEN 0.0
                    ELSE (ST_Length(ir.route_geom::geography) / 1.35) * 0.10
                END AS walk_duration_sec,
                CASE
                    WHEN split_part(ir.travel_day_of_week, '|mode=', 2) = 'CAR' THEN ST_Length(ir.route_geom::geography) / 8.0
                    ELSE 0.0
                END AS car_duration_sec,
                CASE
                    WHEN split_part(ir.travel_day_of_week, '|mode=', 2) = 'PUBLIC_TRANSPORT' THEN ST_Length(ir.route_geom::geography) / 5.0
                    ELSE 0.0
                END AS public_transport_duration_sec,
                CASE WHEN split_part(ir.travel_day_of_week, '|mode=', 2) = 'WALK' THEN ST_Length(ir.route_geom::geography) ELSE 0.0 END AS walk_distance,
                CASE WHEN split_part(ir.travel_day_of_week, '|mode=', 2) = 'PUBLIC_TRANSPORT' THEN 1 ELSE 0 END AS num_transfers
            FROM inserted_routes ir
            CROSS JOIN LATERAL (
                WITH route AS (SELECT ir.route_geom AS geom),
                incidents_24h AS (
                    SELECT i.* FROM incidents_raw i, route r
                    WHERE i.geom IS NOT NULL
                      AND i.incident_datetime >= NOW() - INTERVAL '24 hours'
                      AND ST_DWithin(i.geom::geography, r.geom::geography, 500)
                ), incidents_7d AS (
                    SELECT i.* FROM incidents_raw i, route r
                    WHERE i.geom IS NOT NULL
                      AND i.incident_datetime >= NOW() - INTERVAL '7 days'
                      AND ST_DWithin(i.geom::geography, r.geom::geography, 500)
                ), segments AS (
                    SELECT ST_LineSubstring(r.geom, gs::float8, LEAST((gs + 0.05)::float8, 1.0)) AS geom
                    FROM route r, generate_series(0.0, 0.95, 0.05) AS gs
                ), segment_counts AS (
                    SELECT COUNT(i.row_id) AS incident_count
                    FROM segments s
                    LEFT JOIN incidents_raw i
                      ON i.geom IS NOT NULL
                     AND i.incident_datetime >= NOW() - INTERVAL '24 hours'
                     AND ST_DWithin(i.geom::geography, s.geom::geography, 250)
                    GROUP BY s.geom
                )
                SELECT
                    (SELECT COUNT(*) FROM incidents_24h i, route r WHERE ST_DWithin(i.geom::geography, r.geom::geography, 100))::int AS incidents_near_route_100m_24h,
                    (SELECT COUNT(*) FROM incidents_24h i, route r WHERE ST_DWithin(i.geom::geography, r.geom::geography, 250))::int AS incidents_near_route_250m_24h,
                    (SELECT COUNT(*) FROM incidents_24h)::int AS incidents_near_route_500m_24h,
                    (SELECT COUNT(*) FROM incidents_7d)::int AS incidents_near_route_7d,
                    COALESCE((SELECT AVG(CASE WHEN incident_category ILIKE '%%theft%%' OR incident_category ILIKE '%%larceny%%' THEN 1.0 ELSE 0.0 END) FROM incidents_7d), 0.0) AS theft_ratio_near_route_7d,
                    COALESCE((SELECT AVG(CASE WHEN incident_category ILIKE '%%assault%%' THEN 1.0 ELSE 0.0 END) FROM incidents_7d), 0.0) AS assault_ratio_near_route_7d,
                    COALESCE((SELECT AVG(CASE WHEN EXTRACT(HOUR FROM incident_datetime) >= 20 OR EXTRACT(HOUR FROM incident_datetime) <= 5 THEN 1.0 ELSE 0.0 END) FROM incidents_7d), 0.0) AS night_ratio_near_route_7d,
                    COALESCE((SELECT AVG(ST_Distance(i.geom::geography, r.geom::geography)) FROM incidents_7d i, route r), 9999.0) AS avg_distance_incidents_m,
                    COALESCE((SELECT MAX(incident_count)::double precision FROM segment_counts), 0.0) AS max_segment_density
            ) f
        ), exposure_features AS (
            SELECT *,
                GREATEST(walk_duration_sec + car_duration_sec + public_transport_duration_sec, 1.0) AS total_mode_duration_sec
            FROM features
        ), scored AS (
            SELECT *,
                walk_duration_sec / total_mode_duration_sec AS walk_ratio,
                car_duration_sec / total_mode_duration_sec AS car_ratio,
                public_transport_duration_sec / total_mode_duration_sec AS public_transport_ratio,
                LEAST(1.0, GREATEST(0.0,
                    (1.00 * (walk_duration_sec / total_mode_duration_sec))
                  + (0.45 * (public_transport_duration_sec / total_mode_duration_sec))
                  + (0.18 * (car_duration_sec / total_mode_duration_sec))
                  + LEAST(num_transfers * 0.05, 0.20)
                )) AS mode_exposure_factor,
                LEAST(1.0, GREATEST(0.0,
                    (
                        0.20 * LEAST(incidents_near_route_250m_24h::double precision / 20.0, 1.0)
                      + 0.18 * LEAST(incidents_near_route_7d::double precision / 180.0, 1.0)
                      + 0.17 * LEAST(max_segment_density::double precision / 12.0, 1.0)
                      + 0.12 * (1.0 - LEAST(avg_distance_incidents_m / 500.0, 1.0))
                      + 0.07 * LEAST(night_ratio_near_route_7d, 1.0)
                      + 0.05 * LEAST(theft_ratio_near_route_7d, 1.0)
                      + 0.06 * LEAST(assault_ratio_near_route_7d, 1.0)
                      + 0.08 * travel_hour_pressure
                      + weekend_pressure
                    )
                    *
                    GREATEST(0.25, LEAST(1.45,
                        0.55
                      + 0.75 * LEAST(1.0, GREATEST(0.0,
                            (1.00 * (walk_duration_sec / total_mode_duration_sec))
                          + (0.45 * (public_transport_duration_sec / total_mode_duration_sec))
                          + (0.18 * (car_duration_sec / total_mode_duration_sec))
                          + LEAST(num_transfers * 0.05, 0.20)
                        ))
                      + 0.20 * (walk_duration_sec / total_mode_duration_sec)
                      - 0.18 * (car_duration_sec / total_mode_duration_sec)
                      + 0.08 * LEAST(num_transfers::double precision / 3.0, 1.0)
                    ))
                )) AS target_risk_score
            FROM exposure_features
        ), inserted_features AS (
            INSERT INTO route_risk_features (
                route_id, computed_at, travel_hour, travel_day_of_week,
                incidents_near_route_100m_24h, incidents_near_route_250m_24h,
                incidents_near_route_500m_24h, incidents_near_route_7d,
                theft_ratio_near_route_7d, assault_ratio_near_route_7d,
                night_ratio_near_route_7d, avg_distance_incidents_m, max_segment_density,
                walk_duration_sec, car_duration_sec, public_transport_duration_sec,
                walk_ratio, car_ratio, public_transport_ratio, walk_distance,
                num_transfers, mode_exposure_factor, dominant_transport_mode,
                target_risk_score, target_risk_level
            )
            SELECT
                route_id, NOW(), travel_hour, travel_day_of_week,
                incidents_near_route_100m_24h, incidents_near_route_250m_24h,
                incidents_near_route_500m_24h, incidents_near_route_7d,
                theft_ratio_near_route_7d, assault_ratio_near_route_7d,
                night_ratio_near_route_7d, avg_distance_incidents_m, max_segment_density,
                walk_duration_sec, car_duration_sec, public_transport_duration_sec,
                walk_ratio, car_ratio, public_transport_ratio, walk_distance,
                num_transfers, mode_exposure_factor, dominant_transport_mode,
                target_risk_score,
                CASE
                    WHEN target_risk_score >= 0.75 THEN 'Very High'
                    WHEN target_risk_score >= 0.55 THEN 'High'
                    WHEN target_risk_score >= 0.30 THEN 'Medium'
                    ELSE 'Low'
                END
            FROM scored
            RETURNING route_feature_id, target_risk_score, target_risk_level
        )
        SELECT
            COUNT(*)::int AS inserted_rows,
            MIN(target_risk_score)::double precision AS min_target_risk_score,
            MAX(target_risk_score)::double precision AS max_target_risk_score,
            AVG(target_risk_score)::double precision AS avg_target_risk_score,
            COUNT(*) FILTER (WHERE target_risk_level = 'Low')::int AS low_rows,
            COUNT(*) FILTER (WHERE target_risk_level = 'Medium')::int AS medium_rows,
            COUNT(*) FILTER (WHERE target_risk_level = 'High')::int AS high_rows,
            COUNT(*) FILTER (WHERE target_risk_level = 'Very High')::int AS very_high_rows
        FROM inserted_features;
        """,
        (lookback_days, route_pool_size, route_pool_size, route_pool_size, safe_hours, safe_days, SYNTHETIC_SOURCE, rows_per_bucket),
    )

    result = dict(rows[0]) if rows else {"inserted_rows": 0}
    result.update({
        "status": "ok",
        "source": SYNTHETIC_SOURCE,
        "target_rows_requested": target_rows,
        "lookback_days": lookback_days,
        "hours": safe_hours,
        "days": safe_days,
        "rows_per_bucket": rows_per_bucket,
        "route_pool_size": route_pool_size,
        "before_targeted_rows": before_count,
        "after_targeted_rows": count_training_rows(lookback_days),
    })
    return result


def fetch_route_risk_dataset(lookback_days: int) -> list[dict[str, Any]]:
    return fetch_all_dict(
        """
        SELECT
            rrf.route_feature_id,
            rrf.route_id,
            rrf.computed_at,
            COALESCE(rrf.travel_hour, 12) AS travel_hour,
            COALESCE(NULLIF(split_part(rrf.travel_day_of_week, '|', 1), ''), 'Unknown') AS travel_day_of_week,
            COALESCE(rrf.incidents_near_route_100m_24h, 0) AS incidents_near_route_100m_24h,
            COALESCE(rrf.incidents_near_route_250m_24h, 0) AS incidents_near_route_250m_24h,
            COALESCE(rrf.incidents_near_route_500m_24h, 0) AS incidents_near_route_500m_24h,
            COALESCE(rrf.incidents_near_route_7d, 0) AS incidents_near_route_7d,
            COALESCE(rrf.theft_ratio_near_route_7d, 0) AS theft_ratio_near_route_7d,
            COALESCE(rrf.assault_ratio_near_route_7d, 0) AS assault_ratio_near_route_7d,
            COALESCE(rrf.night_ratio_near_route_7d, 0) AS night_ratio_near_route_7d,
            COALESCE(rrf.avg_distance_incidents_m, 9999) AS avg_distance_incidents_m,
            COALESCE(rrf.max_segment_density, 0) AS max_segment_density,
            COALESCE(rrf.walk_duration_sec, 0) AS walk_duration_sec,
            COALESCE(rrf.car_duration_sec, 0) AS car_duration_sec,
            COALESCE(rrf.public_transport_duration_sec, 0) AS public_transport_duration_sec,
            COALESCE(rrf.walk_ratio, 0) AS walk_ratio,
            COALESCE(rrf.car_ratio, 0) AS car_ratio,
            COALESCE(rrf.public_transport_ratio, 0) AS public_transport_ratio,
            COALESCE(rrf.walk_distance, 0) AS walk_distance,
            COALESCE(rrf.num_transfers, 0) AS num_transfers,
            COALESCE(rrf.mode_exposure_factor, 0) AS mode_exposure_factor,
            COALESCE(rrf.dominant_transport_mode, 'UNKNOWN') AS dominant_transport_mode,
            rrf.target_risk_score,
            rrf.target_risk_level
        FROM route_risk_features rrf
        WHERE rrf.computed_at >= NOW() - (%s::int * INTERVAL '1 day')
          AND rrf.target_risk_score IS NOT NULL
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
    density_250 = minmax(df["incidents_near_route_250m_24h"])
    density_7d = minmax(df["incidents_near_route_7d"])
    max_segment = minmax(df["max_segment_density"])
    avg_distance = pd.to_numeric(df["avg_distance_incidents_m"], errors="coerce").fillna(9999).astype(float)
    distance_pressure = 1.0 - minmax(avg_distance)
    night = pd.to_numeric(df["night_ratio_near_route_7d"], errors="coerce").fillna(0).clip(0, 1)
    theft = pd.to_numeric(df["theft_ratio_near_route_7d"], errors="coerce").fillna(0).clip(0, 1)
    assault = pd.to_numeric(df["assault_ratio_near_route_7d"], errors="coerce").fillna(0).clip(0, 1)
    exposure = pd.to_numeric(df.get("mode_exposure_factor", 0), errors="coerce").fillna(0).clip(0, 1)
    walk_ratio = pd.to_numeric(df.get("walk_ratio", 0), errors="coerce").fillna(0).clip(0, 1)
    car_ratio = pd.to_numeric(df.get("car_ratio", 0), errors="coerce").fillna(0).clip(0, 1)
    transfers = (pd.to_numeric(df.get("num_transfers", 0), errors="coerce").fillna(0) / 3.0).clip(0, 1)

    base_area_score = (
        0.20 * density_250
        + 0.18 * density_7d
        + 0.17 * max_segment
        + 0.12 * distance_pressure
        + 0.07 * night
        + 0.05 * theft
        + 0.06 * assault
    )
    exposure_multiplier = (0.55 + (0.75 * exposure) + (0.20 * walk_ratio) - (0.18 * car_ratio) + (0.08 * transfers)).clip(0.25, 1.45)
    return (base_area_score * exposure_multiplier).clip(0.0, 1.0)

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
    targets = [float(row.get(TARGET_COLUMN) or 0) for row in rows if row.get(TARGET_COLUMN) is not None]
    hours = sorted({int(row.get("travel_hour") or 0) for row in rows})
    levels: dict[str, int] = {}
    for row in rows:
        level = row.get("target_risk_level") or level_from_score(float(row.get(TARGET_COLUMN) or 0))
        levels[str(level)] = levels.get(str(level), 0) + 1
    return {
        "lookback_days": lookback_days,
        "row_count": len(rows),
        "min_computed_at": min(timestamps).isoformat() if timestamps else None,
        "max_computed_at": max(timestamps).isoformat() if timestamps else None,
        "target_min": round(min(targets), 6) if targets else None,
        "target_avg": round(sum(targets) / len(targets), 6) if targets else None,
        "target_max": round(max(targets), 6) if targets else None,
        "target_level_counts": levels,
        "travel_hours": hours,
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

    explicit_target_count = int(pd.to_numeric(df[TARGET_COLUMN], errors="coerce").notna().sum())
    if explicit_target_count >= max(20, int(len(df) * 0.25)):
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0).clip(0, 1)
        label_source = "target_risk_score"
    else:
        df[TARGET_COLUMN] = build_weak_target_scores(df)
        label_source = "weak_labels_from_route_features"

    feature_columns = ROUTE_NUMERIC_FEATURES + ROUTE_CATEGORICAL_FEATURES
    X = df[feature_columns]
    y = df[TARGET_COLUMN].astype(float).clip(0, 1)

    split_index = int(len(df) * (1.0 - test_size))
    split_index = max(1, min(split_index, len(df) - 1))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("regressor", RandomForestRegressor(n_estimators=240, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)),
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
        "risk_level_thresholds": {"medium": 0.30, "high": 0.55, "very_high": 0.75},
    }

    predicted_level_counts: dict[str, int] = {}
    for prediction in predictions:
        level = level_from_score(prediction)
        predicted_level_counts[level] = predicted_level_counts.get(level, 0) + 1

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
        "predicted_level_counts": predicted_level_counts,
        "features": feature_columns,
    }
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
    get_s3_client().put_object(Bucket=MODEL_BUCKET_NAME, Key=key, Body=data, ContentType=content_type)


def upload_artifacts(model_path: Path, metrics_path: Path) -> dict[str, Any]:
    keys = get_artifact_keys()
    upload_bytes_to_model_bucket(keys["model"], model_path.read_bytes(), "application/octet-stream")
    upload_bytes_to_model_bucket(keys["metrics"], metrics_path.read_bytes(), "application/json")
    return {"bucket": MODEL_BUCKET_NAME, "endpoint_url": MODEL_BUCKET_ENDPOINT_URL, "model_key": keys["model"], "metrics_key": keys["metrics"]}


def main() -> dict[str, Any]:
    lookback_days = parse_int_env("ROUTE_RISK_LOOKBACK_DAYS", DEFAULT_LOOKBACK_DAYS, 7, 3650)
    test_size = parse_float_env("ROUTE_RISK_TEST_SIZE", DEFAULT_TEST_SIZE, 0.05, 0.4)
    min_rows = parse_int_env("ROUTE_RISK_MIN_ROWS", DEFAULT_MIN_ROWS, 10, 100000)
    target_rows = parse_int_env("ROUTE_RISK_TARGET_ROWS", DEFAULT_TARGET_ROWS, 0, 100000)
    route_pool_size = parse_int_env("ROUTE_RISK_ROUTE_POOL_SIZE", DEFAULT_ROUTE_POOL_SIZE, 100, 100000)
    hours = parse_csv_int_env("ROUTE_RISK_HOURS", [2, 8, 12, 15, 18, 22])
    days = parse_csv_text_env("ROUTE_RISK_DAYS", ["Wednesday", "Thursday", "Friday", "Saturday"])

    ensure_route_risk_mode_columns()
    purged_rows = purge_synthetic_dataset() if PURGE_SYNTHETIC else 0
    generated_dataset: dict[str, Any] | None = None

    if AUTO_BUILD_DATASET:
        current_rows = count_training_rows(lookback_days)
        rows_needed = max(0, target_rows - current_rows)
        if rows_needed > 0:
            generated_dataset = generate_route_risk_dataset_from_incidents(
                target_rows=rows_needed,
                lookback_days=lookback_days,
                hours=hours,
                days=days,
                route_pool_size=route_pool_size,
            )
        else:
            generated_dataset = {"status": "skipped", "reason": "enough_rows", "current_rows": current_rows, "target_rows": target_rows, "inserted_rows": 0}

    rows = fetch_route_risk_dataset(lookback_days)
    training_result = train_route_risk_model(rows, test_size=test_size, min_rows=min_rows)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{ROUTE_RISK_MODEL_NAME}.joblib"
    metrics_path = MODEL_DIR / f"{ROUTE_RISK_MODEL_NAME}_metrics.json"
    joblib.dump(training_result["bundle"], model_path)

    metrics = dict(training_result["metrics"])
    metrics["dataset"] = summarize_dataset(rows, lookback_days, training_result["label_source"])
    metrics["dataset_generation"] = {
        "auto_build_dataset": AUTO_BUILD_DATASET,
        "purge_synthetic": PURGE_SYNTHETIC,
        "purged_route_requests": purged_rows,
        "target_rows": target_rows,
        "generated": generated_dataset,
    }
    metrics["local_artifacts"] = {"model_path": str(model_path), "metrics_path": str(metrics_path)}
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
        print(json.dumps({"status": "error", "model_name": ROUTE_RISK_MODEL_NAME, "message": str(exc), "failed_at": utc_now_iso()}, indent=2, default=str), flush=True)
        raise
