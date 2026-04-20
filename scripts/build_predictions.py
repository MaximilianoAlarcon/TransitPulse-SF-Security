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
DEFAULT_MIN_HISTORY_POINTS = 2
DEFAULT_STDDEV_MULTIPLIER = 1.0
DEFAULT_RISK_LOOKBACK_HOURS = 6


def floor_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
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
                columns = [d[0] for d in cur.description]
                return [dict(zip(columns, r)) for r in rows]
    finally:
        conn.close()


def fetchone_value(query: str, params: tuple[Any, ...] = ()) -> Any:
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()
                if not row:
                    return None
                if isinstance(row, dict):
                    return next(iter(row.values()))
                return row[0]
    finally:
        conn.close()


def execute_many(query: str, rows: list[dict[str, Any]]) -> None:
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


def get_generated_at() -> datetime:
    return floor_to_hour(datetime.utcnow())


def get_latest_hour(table: str, column: str) -> datetime | None:
    value = fetchone_value(f"SELECT MAX({column}) FROM {table};")
    return floor_to_hour(value) if value else None


def get_forecast_target_hour(base_hour: datetime | None) -> datetime | None:
    if base_hour is None:
        return None
    horizon_hours = int(os.environ.get("FORECAST_HORIZON_HOURS", str(DEFAULT_FORECAST_HORIZON_HOURS)))
    return base_hour + timedelta(hours=horizon_hours)


def fetch_history_strict(target_hour: datetime, history_weeks: int) -> list[dict[str, Any]]:
    history_start = target_hour - timedelta(weeks=history_weeks)
    return fetchall_dicts(
        """
        SELECT bucket_start, police_district, incident_category, total_incidents
        FROM forecast_training_series
        WHERE bucket_start >= %s
          AND bucket_start < %s
          AND EXTRACT(HOUR FROM bucket_start) = EXTRACT(HOUR FROM %s::timestamp)
          AND EXTRACT(DOW FROM bucket_start) = EXTRACT(DOW FROM %s::timestamp)
        ORDER BY bucket_start;
        """,
        (history_start, target_hour, target_hour, target_hour),
    )


def fetch_history_fallback(target_hour: datetime, history_weeks: int) -> list[dict[str, Any]]:
    history_start = target_hour - timedelta(weeks=history_weeks)
    return fetchall_dicts(
        """
        SELECT bucket_start, police_district, incident_category, total_incidents
        FROM forecast_training_series
        WHERE bucket_start >= %s
          AND bucket_start < %s
          AND EXTRACT(HOUR FROM bucket_start) = EXTRACT(HOUR FROM %s::timestamp)
        ORDER BY bucket_start;
        """,
        (history_start, target_hour, target_hour),
    )


def build_forecast_predictions(generated_at: datetime, target_hour: datetime) -> None:
    history_weeks = int(os.environ.get("FORECAST_HISTORY_WEEKS", str(DEFAULT_HISTORY_WEEKS)))
    min_history_points = int(os.environ.get("FORECAST_MIN_HISTORY_POINTS", str(DEFAULT_MIN_HISTORY_POINTS)))
    stddev_multiplier = float(os.environ.get("FORECAST_STDDEV_MULTIPLIER", str(DEFAULT_STDDEV_MULTIPLIER)))

    strict_rows = fetch_history_strict(target_hour, history_weeks)
    fallback_rows = fetch_history_fallback(target_hour, history_weeks)

    strict_grouped: dict[tuple[str, str], list[float]] = {}
    fallback_grouped: dict[tuple[str, str], list[float]] = {}

    for row in strict_rows:
        district = row.get("police_district")
        category = row.get("incident_category")
        if not district or not category:
            continue
        strict_grouped.setdefault((district, category), []).append(safe_float(row.get("total_incidents")))

    for row in fallback_rows:
        district = row.get("police_district")
        category = row.get("incident_category")
        if not district or not category:
            continue
        fallback_grouped.setdefault((district, category), []).append(safe_float(row.get("total_incidents")))

    all_keys = sorted(set(strict_grouped.keys()) | set(fallback_grouped.keys()))
    results: list[dict[str, Any]] = []

    for key in all_keys:
        values = strict_grouped.get(key, [])
        history_mode = "strict"
        if len(values) < min_history_points:
            values = fallback_grouped.get(key, [])
            history_mode = "fallback_hour_only"

        if len(values) < min_history_points:
            continue

        avg = mean(values)
        std = pstdev(values) if len(values) > 1 else 0.0
        lower = max(0.0, avg - (stddev_multiplier * std))
        upper = avg + (stddev_multiplier * std)

        results.append(
            {
                "model_name": FORECAST_MODEL_NAME,
                "generated_at": generated_at,
                "forecast_for": target_hour,
                "police_district": key[0],
                "incident_category": key[1],
                "predicted_incidents": round(avg, 4),
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
            }
        )

    execute_many(
        """
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
        """,
        results,
    )

    print(f"Forecast rows: {len(results)}")


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


def build_anomaly_detections(generated_at: datetime, observed_hour: datetime) -> None:
    observed = fetchall_dicts(
        """
        SELECT police_district, incident_category, SUM(total_incidents) AS observed_value
        FROM incident_counts_hourly
        WHERE bucket_start = %s
        GROUP BY police_district, incident_category
        ORDER BY police_district, incident_category;
        """,
        (observed_hour,),
    )

    history_weeks = int(os.environ.get("FORECAST_HISTORY_WEEKS", str(DEFAULT_HISTORY_WEEKS)))
    min_history_points = int(os.environ.get("FORECAST_MIN_HISTORY_POINTS", str(DEFAULT_MIN_HISTORY_POINTS)))
    stddev_multiplier = float(os.environ.get("FORECAST_STDDEV_MULTIPLIER", str(DEFAULT_STDDEV_MULTIPLIER)))

    strict_rows = fetch_history_strict(observed_hour, history_weeks)
    fallback_rows = fetch_history_fallback(observed_hour, history_weeks)

    strict_grouped: dict[tuple[str, str], list[float]] = {}
    fallback_grouped: dict[tuple[str, str], list[float]] = {}

    for row in strict_rows:
        district = row.get("police_district")
        category = row.get("incident_category")
        if not district or not category:
            continue
        strict_grouped.setdefault((district, category), []).append(safe_float(row.get("total_incidents")))

    for row in fallback_rows:
        district = row.get("police_district")
        category = row.get("incident_category")
        if not district or not category:
            continue
        fallback_grouped.setdefault((district, category), []).append(safe_float(row.get("total_incidents")))

    results: list[dict[str, Any]] = []

    for row in observed:
        district = row.get("police_district")
        category = row.get("incident_category")
        if not district or not category:
            continue

        key = (district, category)
        values = strict_grouped.get(key, [])
        if len(values) < min_history_points:
            values = fallback_grouped.get(key, [])

        if len(values) < min_history_points:
            continue

        avg = mean(values)
        std = pstdev(values) if len(values) > 1 else 0.0
        lower = max(0.0, avg - (stddev_multiplier * std))
        upper = avg + (stddev_multiplier * std)
        observed_value = safe_float(row.get("observed_value"))
        anomaly = observed_value < lower or observed_value > upper

        results.append(
            {
                "model_name": ANOMALY_MODEL_NAME,
                "generated_at": generated_at,
                "bucket_start": observed_hour,
                "police_district": district,
                "incident_category": category,
                "expected_min": round(lower, 4),
                "expected_max": round(upper, 4),
                "observed_value": round(observed_value, 4),
                "anomaly": anomaly,
                "severity": compute_anomaly_severity(observed_value, lower, upper),
            }
        )

    execute_many(
        """
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
        """,
        results,
    )

    print(f"Anomaly rows: {len(results)}")


def fetch_recent_risk_features(target_timestamp: datetime) -> list[dict[str, Any]]:
    return fetchall_dicts(
        """
        SELECT
            feature_timestamp,
            police_district,
            incident_category,
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
        """,
        (target_timestamp,),
    )


def fetch_recent_risk_baselines(target_timestamp: datetime, lookback_hours: int) -> list[dict[str, Any]]:
    start_ts = target_timestamp - timedelta(hours=lookback_hours)
    return fetchall_dicts(
        """
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
        """,
        (start_ts, target_timestamp),
    )


def fetch_forecast_map_for_risk(target_timestamp: datetime) -> dict[tuple[str, str], dict[str, Any]]:
    rows = fetchall_dicts(
        """
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
        """,
        (FORECAST_MODEL_NAME, target_timestamp),
    )

    output: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        district = row.get("police_district")
        category = row.get("incident_category")
        if district and category:
            output[(district, category)] = row
    return output


def normalize_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return clamp(numerator / denominator, 0.0, 1.0)


def compute_risk_score(feature: dict[str, Any], baseline: dict[str, Any] | None, forecast_row: dict[str, Any] | None) -> float:
    incidents_last_1h = safe_float(feature.get("incidents_last_1h"))
    incidents_last_24h = safe_float(feature.get("incidents_last_24h"))
    incidents_last_7d = safe_float(feature.get("incidents_last_7d"))
    open_ratio = clamp(safe_float(feature.get("open_active_ratio_24h")), 0.0, 1.0)
    online_ratio = clamp(safe_float(feature.get("filed_online_ratio_24h")), 0.0, 1.0)
    avg_delay = max(0.0, safe_float(feature.get("avg_report_delay_minutes_24h")))

    max_1h = max(1.0, safe_float((baseline or {}).get("max_incidents_last_1h"), 1.0))
    max_24h = max(1.0, safe_float((baseline or {}).get("max_incidents_last_24h"), 1.0))
    max_7d = max(1.0, safe_float((baseline or {}).get("max_incidents_last_7d"), 1.0))
    max_delay = max(1.0, safe_float((baseline or {}).get("max_avg_report_delay_minutes_24h"), 1.0))

    recent_1h_component = normalize_ratio(incidents_last_1h, max_1h)
    recent_24h_component = normalize_ratio(incidents_last_24h, max_24h)
    recent_7d_component = normalize_ratio(incidents_last_7d, max_7d)
    delay_component = normalize_ratio(avg_delay, max_delay)

    forecast_component = 0.0
    if forecast_row:
        predicted_incidents = safe_float(forecast_row.get("predicted_incidents"))
        forecast_upper = max(1.0, safe_float(forecast_row.get("upper_bound"), predicted_incidents))
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


def build_risk_predictions(generated_at: datetime, target_timestamp: datetime) -> None:
    lookback_hours = int(os.environ.get("RISK_BASELINE_LOOKBACK_HOURS", str(DEFAULT_RISK_LOOKBACK_HOURS)))

    feature_rows = fetch_recent_risk_features(target_timestamp)
    baseline_rows = fetch_recent_risk_baselines(target_timestamp, lookback_hours)
    forecast_map = fetch_forecast_map_for_risk(target_timestamp)

    baseline_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in baseline_rows:
        district = row.get("police_district")
        category = row.get("incident_category")
        if district and category:
            baseline_map[(district, category)] = row

    results: list[dict[str, Any]] = []

    for row in feature_rows:
        district = row.get("police_district")
        category = row.get("incident_category")
        if not district or not category:
            continue

        key = (district, category)
        score = compute_risk_score(row, baseline_map.get(key), forecast_map.get(key))

        results.append(
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

    execute_many(
        """
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
        """,
        results,
    )

    print(f"Risk rows: {len(results)}")


def main() -> None:
    generated_at = get_generated_at()

    observed_hour = get_latest_hour("incident_counts_hourly", "bucket_start")
    risk_hour = get_latest_hour("risk_features_hourly", "feature_timestamp")
    forecast_base = get_latest_hour("forecast_training_series", "bucket_start")
    forecast_target = get_forecast_target_hour(forecast_base)

    print("===== START build_predictions =====")
    print(f"generated_at={generated_at}")
    print(f"observed_hour={observed_hour}")
    print(f"risk_hour={risk_hour}")
    print(f"forecast_target={forecast_target}")

    if forecast_target:
        build_forecast_predictions(generated_at, forecast_target)
    else:
        print("Forecast rows: 0 (no forecast base hour found)")

    if observed_hour:
        build_anomaly_detections(generated_at, observed_hour)
    else:
        print("Anomaly rows: 0 (no observed hour found)")

    if risk_hour:
        build_risk_predictions(generated_at, risk_hour)
    else:
        print("Risk rows: 0 (no risk hour found)")

    print("===== DONE =====")


if __name__ == "__main__":
    main()
