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
DEFAULT_MIN_HISTORY_POINTS = 4
DEFAULT_STDDEV_MULTIPLIER = 1.5
DEFAULT_RISK_LOOKBACK_HOURS = 6


def floor_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fetchall_dicts(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

                if not rows:
                    return []

                if isinstance(rows[0], dict):
                    return rows

                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()


def execute_many(query: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                for row in rows:
                    cur.execute(query, row)
    finally:
        conn.close()


def get_generated_at() -> datetime:
    return floor_to_hour(datetime.utcnow())


def get_forecast_target_hour(generated_at: datetime) -> datetime:
    horizon_hours = int(
        os.environ.get("FORECAST_HORIZON_HOURS", str(DEFAULT_FORECAST_HORIZON_HOURS))
    )
    return generated_at + timedelta(hours=horizon_hours)


def get_anomaly_observed_hour(generated_at: datetime) -> datetime:
    return generated_at


def get_risk_target_timestamp(generated_at: datetime) -> datetime:
    return generated_at


def fetch_forecast_history(
    target_hour: datetime,
    history_weeks: int,
) -> list[dict[str, Any]]:
    history_start = target_hour - timedelta(weeks=history_weeks + 1)

    query = """
    SELECT
        bucket_start,
        police_district,
        incident_category,
        total_incidents
    FROM forecast_training_series
    WHERE bucket_start >= %s
      AND bucket_start < %s
      AND EXTRACT(HOUR FROM bucket_start) = EXTRACT(HOUR FROM %s::timestamp)
      AND EXTRACT(DOW FROM bucket_start) = EXTRACT(DOW FROM %s::timestamp)
    ORDER BY police_district, incident_category, bucket_start;
    """
    return fetchall_dicts(
        query,
        (history_start, target_hour, target_hour, target_hour),
    )


def build_forecast_predictions(
    generated_at: datetime,
    target_hour: datetime,
) -> None:
    history_weeks = int(
        os.environ.get("FORECAST_HISTORY_WEEKS", str(DEFAULT_HISTORY_WEEKS))
    )
    min_history_points = int(
        os.environ.get("FORECAST_MIN_HISTORY_POINTS", str(DEFAULT_MIN_HISTORY_POINTS))
    )
    stddev_multiplier = float(
        os.environ.get("FORECAST_STDDEV_MULTIPLIER", str(DEFAULT_STDDEV_MULTIPLIER))
    )

    rows = fetch_forecast_history(target_hour=target_hour, history_weeks=history_weeks)

    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        district = row.get("police_district")
        category = row.get("incident_category")
        total_incidents = safe_float(row.get("total_incidents"))
        if not district or not category:
            continue
        key = (district, category)
        grouped.setdefault(key, []).append(total_incidents)

    upsert_rows: list[dict[str, Any]] = []

    for (district, category), values in grouped.items():
        if len(values) < min_history_points:
            continue

        predicted = mean(values)
        stddev = pstdev(values) if len(values) > 1 else 0.0

        lower_bound = max(0.0, predicted - stddev_multiplier * stddev)
        upper_bound = max(predicted, predicted + stddev_multiplier * stddev)

        upsert_rows.append(
            {
                "model_name": FORECAST_MODEL_NAME,
                "generated_at": generated_at,
                "forecast_for": target_hour,
                "police_district": district,
                "incident_category": category,
                "predicted_incidents": round(predicted, 4),
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
            }
        )

    upsert_sql = """
    INSERT INTO forecast_predictions (
        model_name,
        generated_at,
        forecast_for,
        police_district,
        incident_category,
        predicted_incidents,
        lower_bound,
        upper_bound
    )
    VALUES (
        %(model_name)s,
        %(generated_at)s,
        %(forecast_for)s,
        %(police_district)s,
        %(incident_category)s,
        %(predicted_incidents)s,
        %(lower_bound)s,
        %(upper_bound)s
    )
    ON CONFLICT (model_name, forecast_for, police_district, incident_category)
    DO UPDATE SET
        generated_at = EXCLUDED.generated_at,
        predicted_incidents = EXCLUDED.predicted_incidents,
        lower_bound = EXCLUDED.lower_bound,
        upper_bound = EXCLUDED.upper_bound;
    """

    execute_many(upsert_sql, upsert_rows)
    print(
        f"Upserted {len(upsert_rows)} rows into forecast_predictions for {target_hour}."
    )


def fetch_observed_counts(observed_hour: datetime) -> list[dict[str, Any]]:
    query = """
    SELECT
        bucket_start,
        police_district,
        incident_category,
        SUM(total_incidents) AS observed_value
    FROM incident_counts_hourly
    WHERE bucket_start = %s
    GROUP BY bucket_start, police_district, incident_category
    ORDER BY police_district, incident_category;
    """
    return fetchall_dicts(query, (observed_hour,))


def fetch_forecast_predictions_for_hour(target_hour: datetime) -> list[dict[str, Any]]:
    query = """
    SELECT
        forecast_for,
        police_district,
        incident_category,
        predicted_incidents,
        lower_bound,
        upper_bound
    FROM forecast_predictions
    WHERE model_name = %s
      AND forecast_for = %s;
    """
    return fetchall_dicts(query, (FORECAST_MODEL_NAME, target_hour))


def compute_anomaly_severity(observed: float, lower: float, upper: float) -> str:
    if lower <= observed <= upper:
        return "normal"

    if observed > upper:
        reference = max(upper, 1.0)
        deviation_ratio = (observed - upper) / reference
    else:
        reference = max(lower, 1.0)
        deviation_ratio = (lower - observed) / reference

    if deviation_ratio >= 0.75:
        return "high"
    if deviation_ratio >= 0.25:
        return "medium"
    return "low"


def build_anomaly_detections(
    generated_at: datetime,
    observed_hour: datetime,
) -> None:
    observed_rows = fetch_observed_counts(observed_hour)
    forecast_rows = fetch_forecast_predictions_for_hour(observed_hour)

    forecast_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in forecast_rows:
        district = row.get("police_district")
        category = row.get("incident_category")
        if not district or not category:
            continue
        forecast_map[(district, category)] = row

    upsert_rows: list[dict[str, Any]] = []

    for obs in observed_rows:
        district = obs.get("police_district")
        category = obs.get("incident_category")
        if not district or not category:
            continue

        observed_value = safe_float(obs.get("observed_value"))
        forecast = forecast_map.get((district, category))
        if not forecast:
            continue

        expected_min = safe_float(forecast.get("lower_bound"))
        expected_max = safe_float(forecast.get("upper_bound"))
        is_anomaly = observed_value < expected_min or observed_value > expected_max
        severity = compute_anomaly_severity(observed_value, expected_min, expected_max)

        upsert_rows.append(
            {
                "model_name": ANOMALY_MODEL_NAME,
                "generated_at": generated_at,
                "bucket_start": observed_hour,
                "police_district": district,
                "incident_category": category,
                "expected_min": round(expected_min, 4),
                "expected_max": round(expected_max, 4),
                "observed_value": round(observed_value, 4),
                "anomaly": is_anomaly,
                "severity": severity,
            }
        )

    upsert_sql = """
    INSERT INTO anomaly_detections (
        model_name,
        generated_at,
        bucket_start,
        police_district,
        incident_category,
        expected_min,
        expected_max,
        observed_value,
        anomaly,
        severity
    )
    VALUES (
        %(model_name)s,
        %(generated_at)s,
        %(bucket_start)s,
        %(police_district)s,
        %(incident_category)s,
        %(expected_min)s,
        %(expected_max)s,
        %(observed_value)s,
        %(anomaly)s,
        %(severity)s
    )
    ON CONFLICT (model_name, bucket_start, police_district, incident_category)
    DO UPDATE SET
        generated_at = EXCLUDED.generated_at,
        expected_min = EXCLUDED.expected_min,
        expected_max = EXCLUDED.expected_max,
        observed_value = EXCLUDED.observed_value,
        anomaly = EXCLUDED.anomaly,
        severity = EXCLUDED.severity;
    """

    execute_many(upsert_sql, upsert_rows)
    print(
        f"Upserted {len(upsert_rows)} rows into anomaly_detections for {observed_hour}."
    )


def fetch_recent_risk_features(target_timestamp: datetime) -> list[dict[str, Any]]:
    query = """
    SELECT
        feature_timestamp,
        police_district,
        incident_category,
        hour_of_day,
        day_of_week,
        month_of_year,
        incidents_last_1h,
        incidents_last_3h,
        incidents_last_6h,
        incidents_last_24h,
        incidents_last_7d,
        open_active_ratio_24h,
        filed_online_ratio_24h,
        avg_report_delay_minutes_24h
    FROM risk_features_hourly
    WHERE feature_timestamp = %s
    ORDER BY police_district, incident_category;
    """
    return fetchall_dicts(query, (target_timestamp,))


def fetch_recent_risk_baselines(
    target_timestamp: datetime,
    lookback_hours: int,
) -> list[dict[str, Any]]:
    start_ts = target_timestamp - timedelta(hours=lookback_hours)

    query = """
    SELECT
        police_district,
        incident_category,
        MAX(incidents_last_1h) AS max_incidents_last_1h,
        MAX(incidents_last_24h) AS max_incidents_last_24h,
        MAX(incidents_last_7d) AS max_incidents_last_7d,
        MAX(avg_report_delay_minutes_24h) AS max_avg_report_delay_minutes_24h
    FROM risk_features_hourly
    WHERE feature_timestamp >= %s
      AND feature_timestamp <= %s
    GROUP BY police_district, incident_category;
    """
    return fetchall_dicts(query, (start_ts, target_timestamp))


def fetch_forecast_map_for_risk(
    target_timestamp: datetime,
) -> dict[tuple[str, str], dict[str, Any]]:
    rows = fetch_forecast_predictions_for_hour(target_timestamp)
    output: dict[tuple[str, str], dict[str, Any]] = {}

    for row in rows:
        district = row.get("police_district")
        category = row.get("incident_category")
        if not district or not category:
            continue
        output[(district, category)] = row

    return output


def normalize_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return clamp(numerator / denominator, 0.0, 1.0)


def compute_risk_score(
    feature: dict[str, Any],
    baseline: dict[str, Any] | None,
    forecast_row: dict[str, Any] | None,
) -> float:
    incidents_last_1h = safe_float(feature.get("incidents_last_1h"))
    incidents_last_24h = safe_float(feature.get("incidents_last_24h"))
    incidents_last_7d = safe_float(feature.get("incidents_last_7d"))
    open_ratio = clamp(safe_float(feature.get("open_active_ratio_24h")), 0.0, 1.0)
    online_ratio = clamp(safe_float(feature.get("filed_online_ratio_24h")), 0.0, 1.0)
    avg_delay = max(0.0, safe_float(feature.get("avg_report_delay_minutes_24h")))

    max_1h = max(1.0, safe_float((baseline or {}).get("max_incidents_last_1h"), 1.0))
    max_24h = max(1.0, safe_float((baseline or {}).get("max_incidents_last_24h"), 1.0))
    max_7d = max(1.0, safe_float((baseline or {}).get("max_incidents_last_7d"), 1.0))
    max_delay = max(
        1.0,
        safe_float((baseline or {}).get("max_avg_report_delay_minutes_24h"), 1.0),
    )

    recent_1h_component = normalize_ratio(incidents_last_1h, max_1h)
    recent_24h_component = normalize_ratio(incidents_last_24h, max_24h)
    recent_7d_component = normalize_ratio(incidents_last_7d, max_7d)
    delay_component = normalize_ratio(avg_delay, max_delay)

    forecast_component = 0.0
    if forecast_row:
        predicted_incidents = safe_float(forecast_row.get("predicted_incidents"))
        forecast_upper = max(
            1.0,
            safe_float(forecast_row.get("upper_bound"), predicted_incidents),
        )
        forecast_component = normalize_ratio(predicted_incidents, forecast_upper)

    raw_score = (
        0.30 * recent_1h_component
        + 0.25 * recent_24h_component
        + 0.20 * recent_7d_component
        + 0.15 * open_ratio
        + 0.05 * online_ratio
        + 0.03 * delay_component
        + 0.02 * forecast_component
    )

    return round(clamp(raw_score * 100.0, 0.0, 100.0), 4)


def map_risk_level(score: float) -> str:
    if score >= 75:
        return "Very High"
    if score >= 50:
        return "High"
    if score >= 25:
        return "Medium"
    return "Low"


def build_risk_predictions(
    generated_at: datetime,
    target_timestamp: datetime,
) -> None:
    lookback_hours = int(
        os.environ.get("RISK_BASELINE_LOOKBACK_HOURS", str(DEFAULT_RISK_LOOKBACK_HOURS))
    )

    feature_rows = fetch_recent_risk_features(target_timestamp)
    baseline_rows = fetch_recent_risk_baselines(target_timestamp, lookback_hours)
    forecast_map = fetch_forecast_map_for_risk(target_timestamp)

    baseline_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in baseline_rows:
        district = row.get("police_district")
        category = row.get("incident_category")
        if not district or not category:
            continue
        baseline_map[(district, category)] = row

    upsert_rows: list[dict[str, Any]] = []

    for row in feature_rows:
        district = row.get("police_district")
        category = row.get("incident_category")
        if not district or not category:
            continue

        key = (district, category)
        score = compute_risk_score(
            feature=row,
            baseline=baseline_map.get(key),
            forecast_row=forecast_map.get(key),
        )

        upsert_rows.append(
            {
                "model_name": RISK_MODEL_NAME,
                "generated_at": generated_at,
                "target_timestamp": target_timestamp,
                "police_district": district,
                "incident_category": category,
                "risk_score": score,
                "risk_level": map_risk_level(score),
            }
        )

    upsert_sql = """
    INSERT INTO risk_predictions (
        model_name,
        generated_at,
        target_timestamp,
        police_district,
        incident_category,
        risk_score,
        risk_level
    )
    VALUES (
        %(model_name)s,
        %(generated_at)s,
        %(target_timestamp)s,
        %(police_district)s,
        %(incident_category)s,
        %(risk_score)s,
        %(risk_level)s
    )
    ON CONFLICT (model_name, target_timestamp, police_district, incident_category)
    DO UPDATE SET
        generated_at = EXCLUDED.generated_at,
        risk_score = EXCLUDED.risk_score,
        risk_level = EXCLUDED.risk_level;
    """

    execute_many(upsert_sql, upsert_rows)
    print(f"Upserted {len(upsert_rows)} rows into risk_predictions for {target_timestamp}.")


def main() -> None:
    generated_at = get_generated_at()
    forecast_target_hour = get_forecast_target_hour(generated_at)
    anomaly_observed_hour = get_anomaly_observed_hour(generated_at)
    risk_target_timestamp = get_risk_target_timestamp(generated_at)

    print("===== START build_predictions =====")
    print(f"generated_at={generated_at}")
    print(f"forecast_target_hour={forecast_target_hour}")
    print(f"anomaly_observed_hour={anomaly_observed_hour}")
    print(f"risk_target_timestamp={risk_target_timestamp}")

    build_forecast_predictions(
        generated_at=generated_at,
        target_hour=forecast_target_hour,
    )

    build_anomaly_detections(
        generated_at=generated_at,
        observed_hour=anomaly_observed_hour,
    )

    build_risk_predictions(
        generated_at=generated_at,
        target_timestamp=risk_target_timestamp,
    )

    print("===== build_predictions COMPLETED =====")


if __name__ == "__main__":
    main()