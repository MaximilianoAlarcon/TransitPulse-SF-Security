# ===== CAMBIOS CLAVE =====
# - Forecast con fallback (sin depender estrictamente de DOW)
# - Anomalías ya no dependen de forecast_predictions
# - DEFAULT_MIN_HISTORY_POINTS reducido a 2

import os
import sys
from datetime import datetime, timedelta
from statistics import mean, pstdev
from typing import Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import get_db_connection

FORECAST_MODEL_NAME = "baseline_seasonal_v1"
ANOMALY_MODEL_NAME = "range_detector_v1"
RISK_MODEL_NAME = "heuristic_risk_v1"

DEFAULT_FORECAST_HORIZON_HOURS = 1
DEFAULT_HISTORY_WEEKS = 8
DEFAULT_MIN_HISTORY_POINTS = 2   # 🔥 cambiado
DEFAULT_STDDEV_MULTIPLIER = 1.5
DEFAULT_RISK_LOOKBACK_HOURS = 6


def floor_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except:
        return default


def fetchall_dicts(query, params=()):
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                if not rows:
                    return []
                columns = [d[0] for d in cur.description]
                return [dict(zip(columns, r)) for r in rows]
    finally:
        conn.close()


def fetchone_value(query):
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(query)
                row = cur.fetchone()
                return row[0] if row else None
    finally:
        conn.close()


def execute_many(query, rows):
    if not rows:
        return
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                for r in rows:
                    cur.execute(query, r)
    finally:
        conn.close()


# ===============================
# TIME HELPERS
# ===============================

def get_generated_at():
    return floor_to_hour(datetime.utcnow())


def get_latest_hour(table, column):
    return fetchone_value(f"SELECT MAX({column}) FROM {table};")


# ===============================
# FORECAST
# ===============================

def fetch_history_strict(target_hour, history_weeks):
    return fetchall_dicts("""
        SELECT bucket_start, police_district, incident_category, total_incidents
        FROM forecast_training_series
        WHERE bucket_start >= %s
          AND bucket_start < %s
          AND EXTRACT(HOUR FROM bucket_start) = EXTRACT(HOUR FROM %s)
          AND EXTRACT(DOW FROM bucket_start) = EXTRACT(DOW FROM %s)
    """, (target_hour - timedelta(weeks=history_weeks), target_hour, target_hour, target_hour))


def fetch_history_fallback(target_hour, history_weeks):
    return fetchall_dicts("""
        SELECT bucket_start, police_district, incident_category, total_incidents
        FROM forecast_training_series
        WHERE bucket_start >= %s
          AND bucket_start < %s
          AND EXTRACT(HOUR FROM bucket_start) = EXTRACT(HOUR FROM %s)
    """, (target_hour - timedelta(weeks=history_weeks), target_hour, target_hour))


def build_forecast_predictions(generated_at, target_hour):
    history_weeks = DEFAULT_HISTORY_WEEKS

    rows = fetch_history_strict(target_hour, history_weeks)

    if not rows:
        print("⚠️ fallback forecast (sin DOW)")
        rows = fetch_history_fallback(target_hour, history_weeks)

    grouped = {}
    for r in rows:
        k = (r["police_district"], r["incident_category"])
        grouped.setdefault(k, []).append(safe_float(r["total_incidents"]))

    results = []

    for (d, c), vals in grouped.items():
        if len(vals) < DEFAULT_MIN_HISTORY_POINTS:
            continue

        avg = mean(vals)
        std = pstdev(vals) if len(vals) > 1 else 0

        results.append({
            "model_name": FORECAST_MODEL_NAME,
            "generated_at": generated_at,
            "forecast_for": target_hour,
            "police_district": d,
            "incident_category": c,
            "predicted_incidents": avg,
            "lower_bound": max(0, avg - std),
            "upper_bound": avg + std
        })

    execute_many("""
        INSERT INTO forecast_predictions (...)
        VALUES (...)
        ON CONFLICT (...) DO UPDATE SET ...
    """, results)

    print(f"Forecast rows: {len(results)}")


# ===============================
# ANOMALÍAS (🔥 CAMBIO IMPORTANTE)
# ===============================

def build_anomaly_detections(generated_at, observed_hour):
    observed = fetchall_dicts("""
        SELECT police_district, incident_category, SUM(total_incidents) AS val
        FROM incident_counts_hourly
        WHERE bucket_start = %s
        GROUP BY police_district, incident_category
    """, (observed_hour,))

    history = fetch_history_fallback(observed_hour, DEFAULT_HISTORY_WEEKS)

    grouped = {}
    for r in history:
        k = (r["police_district"], r["incident_category"])
        grouped.setdefault(k, []).append(safe_float(r["total_incidents"]))

    results = []

    for row in observed:
        key = (row["police_district"], row["incident_category"])
        vals = grouped.get(key, [])

        if len(vals) < 2:
            continue

        avg = mean(vals)
        std = pstdev(vals) if len(vals) > 1 else 0

        lower = max(0, avg - std)
        upper = avg + std

        val = safe_float(row["val"])
        anomaly = val < lower or val > upper

        results.append({
            "model_name": ANOMALY_MODEL_NAME,
            "generated_at": generated_at,
            "bucket_start": observed_hour,
            "police_district": key[0],
            "incident_category": key[1],
            "expected_min": lower,
            "expected_max": upper,
            "observed_value": val,
            "anomaly": anomaly,
            "severity": "high" if anomaly else "normal"
        })

    execute_many("""
        INSERT INTO anomaly_detections (...)
        VALUES (...)
        ON CONFLICT (...) DO UPDATE SET ...
    """, results)

    print(f"Anomaly rows: {len(results)}")


# ===============================
# MAIN
# ===============================

def main():
    generated_at = get_generated_at()

    observed_hour = get_latest_hour("incident_counts_hourly", "bucket_start")
    risk_hour = get_latest_hour("risk_features_hourly", "feature_timestamp")
    forecast_base = get_latest_hour("forecast_training_series", "bucket_start")

    forecast_target = forecast_base + timedelta(hours=1) if forecast_base else None

    print("===== START build_predictions =====")
    print(generated_at, observed_hour, risk_hour, forecast_target)

    if forecast_target:
        build_forecast_predictions(generated_at, forecast_target)

    if observed_hour:
        build_anomaly_detections(generated_at, observed_hour)

    print("===== DONE =====")


if __name__ == "__main__":
    main()