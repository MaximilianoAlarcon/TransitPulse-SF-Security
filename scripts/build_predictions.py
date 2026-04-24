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

# Nueva version del modelo de riesgo.
# Mantener v2 permite comparar resultados contra heuristic_risk_v1 sin pisarlos.
RISK_MODEL_NAME = os.environ.get("RISK_MODEL_NAME", "heuristic_risk_v2")

DEFAULT_FORECAST_HORIZON_HOURS = 1
DEFAULT_HISTORY_WEEKS = 12
DEFAULT_MIN_HISTORY_POINTS = 2
DEFAULT_STDDEV_MULTIPLIER = 1.0

# Riesgo: usar mas historia que la version anterior.
DEFAULT_RISK_HISTORY_WEEKS = 12
DEFAULT_RISK_MIN_HISTORY_POINTS = 4
DEFAULT_RISK_STDDEV_FLOOR = 1.0
DEFAULT_RISK_Z_CAP = 3.0

RISK_FEATURE_COLUMNS = [
    "incidents_last_1h",
    "incidents_last_3h",
    "incidents_last_6h",
    "incidents_last_24h",
    "incidents_last_7d",
    "avg_report_delay_minutes_24h",
]


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
        if len(values) < min_history_points:
            values = fallback_grouped.get(key, [])

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


def build_groupings(rows: list[dict[str, Any]]) -> dict[str, dict[Any, list[float]]]:
    groupings = {
        "district_category": {},
        "category": {},
        "global": {},
    }

    for row in rows:
        district = row.get("police_district")
        category = row.get("incident_category")
        total_incidents = safe_float(row.get("total_incidents"))

        if district and category:
            groupings["district_category"].setdefault((district, category), []).append(total_incidents)

        if category:
            groupings["category"].setdefault(category, []).append(total_incidents)

        groupings["global"].setdefault("all", []).append(total_incidents)

    return groupings


def select_anomaly_history(
    district: str,
    category: str,
    strict_groupings: dict[str, dict[Any, list[float]]],
    fallback_groupings: dict[str, dict[Any, list[float]]],
    min_history_points: int,
) -> tuple[list[float], str]:
    candidates = [
        ("district_category_strict", strict_groupings["district_category"].get((district, category), [])),
        ("district_category_fallback", fallback_groupings["district_category"].get((district, category), [])),
        ("category_strict", strict_groupings["category"].get(category, [])),
        ("category_fallback", fallback_groupings["category"].get(category, [])),
        ("global_strict", strict_groupings["global"].get("all", [])),
        ("global_fallback", fallback_groupings["global"].get("all", [])),
    ]

    for mode, values in candidates:
        if len(values) >= min_history_points:
            return values, mode

    return [], "insufficient_history"


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

    strict_groupings = build_groupings(strict_rows)
    fallback_groupings = build_groupings(fallback_rows)

    results: list[dict[str, Any]] = []

    for row in observed:
        district = row.get("police_district")
        category = row.get("incident_category")
        if not district or not category:
            continue

        values, history_mode = select_anomaly_history(
            district=district,
            category=category,
            strict_groupings=strict_groupings,
            fallback_groupings=fallback_groupings,
            min_history_points=min_history_points,
        )

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


def fetch_risk_history_strict(target_timestamp: datetime, history_weeks: int) -> list[dict[str, Any]]:
    history_start = target_timestamp - timedelta(weeks=history_weeks)
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
            avg_report_delay_minutes_24h
        FROM risk_features_hourly
        WHERE feature_timestamp >= %s
          AND feature_timestamp < %s
          AND EXTRACT(HOUR FROM feature_timestamp) = EXTRACT(HOUR FROM %s::timestamp)
          AND EXTRACT(DOW FROM feature_timestamp) = EXTRACT(DOW FROM %s::timestamp)
        ORDER BY feature_timestamp;
        """,
        (history_start, target_timestamp, target_timestamp, target_timestamp),
    )


def fetch_risk_history_fallback(target_timestamp: datetime, history_weeks: int) -> list[dict[str, Any]]:
    history_start = target_timestamp - timedelta(weeks=history_weeks)
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
            avg_report_delay_minutes_24h
        FROM risk_features_hourly
        WHERE feature_timestamp >= %s
          AND feature_timestamp < %s
          AND EXTRACT(HOUR FROM feature_timestamp) = EXTRACT(HOUR FROM %s::timestamp)
        ORDER BY feature_timestamp;
        """,
        (history_start, target_timestamp, target_timestamp),
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


def fetch_anomaly_map_for_risk(target_timestamp: datetime) -> dict[tuple[str, str], dict[str, Any]]:
    rows = fetchall_dicts(
        """
        SELECT
            bucket_start,
            police_district,
            incident_category,
            anomaly,
            severity
        FROM anomaly_detections
        WHERE model_name = %s
          AND bucket_start = %s;
        """,
        (ANOMALY_MODEL_NAME, target_timestamp),
    )

    output: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        district = row.get("police_district")
        category = row.get("incident_category")
        if district and category:
            output[(district, category)] = row
    return output


def build_risk_groupings(rows: list[dict[str, Any]]) -> dict[str, dict[Any, list[dict[str, Any]]]]:
    groupings: dict[str, dict[Any, list[dict[str, Any]]]] = {
        "district_category": {},
        "category": {},
        "global": {},
    }

    for row in rows:
        district = row.get("police_district")
        category = row.get("incident_category")

        if district and category:
            groupings["district_category"].setdefault((district, category), []).append(row)

        if category:
            groupings["category"].setdefault(category, []).append(row)

        groupings["global"].setdefault("all", []).append(row)

    return groupings


def values_for_metric(rows: list[dict[str, Any]], metric: str) -> list[float]:
    return [safe_float(row.get(metric)) for row in rows]


def compute_metric_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}

    for metric in RISK_FEATURE_COLUMNS:
        values = values_for_metric(rows, metric)
        avg = mean(values) if values else 0.0
        std = pstdev(values) if len(values) > 1 else 0.0

        stats[metric] = {
            "avg": avg,
            "std": std,
            "count": float(len(values)),
        }

    return stats


def select_risk_baseline_stats(
    district: str,
    category: str,
    strict_groupings: dict[str, dict[Any, list[dict[str, Any]]]],
    fallback_groupings: dict[str, dict[Any, list[dict[str, Any]]]],
    min_history_points: int,
) -> tuple[dict[str, dict[str, float]], str]:
    candidates = [
        ("district_category_strict", strict_groupings["district_category"].get((district, category), [])),
        ("district_category_fallback", fallback_groupings["district_category"].get((district, category), [])),
        ("category_strict", strict_groupings["category"].get(category, [])),
        ("category_fallback", fallback_groupings["category"].get(category, [])),
        ("global_strict", strict_groupings["global"].get("all", [])),
        ("global_fallback", fallback_groupings["global"].get("all", [])),
    ]

    for mode, rows in candidates:
        if len(rows) >= min_history_points:
            return compute_metric_stats(rows), mode

    return {}, "insufficient_history"


def normalize_metric_against_history(value: float, stats: dict[str, float]) -> float:
    """
    Normalizacion calibrada:
    - value / (avg + 2*std) conserva la idea de volumen absoluto relativo al historial.
    - z-score premia valores inusualmente altos sin saturar tan rapido como value / max.
    - Si std es 0 o muy bajo, se usa un piso para evitar divisiones agresivas.
    """
    avg = max(0.0, safe_float(stats.get("avg")))
    std_floor = float(os.environ.get("RISK_STDDEV_FLOOR", str(DEFAULT_RISK_STDDEV_FLOOR)))
    z_cap = float(os.environ.get("RISK_Z_CAP", str(DEFAULT_RISK_Z_CAP)))
    std = max(std_floor, safe_float(stats.get("std")))

    expected_high = max(1.0, avg + (2.0 * std))
    level_component = clamp(value / expected_high, 0.0, 1.0)

    z_score = max(0.0, (value - avg) / std)
    z_component = clamp(z_score / z_cap, 0.0, 1.0)

    return clamp((0.65 * level_component) + (0.35 * z_component), 0.0, 1.0)


def normalize_forecast_component(forecast_row: dict[str, Any] | None) -> float:
    if not forecast_row:
        return 0.0

    predicted_incidents = safe_float(forecast_row.get("predicted_incidents"))
    lower_bound = safe_float(forecast_row.get("lower_bound"))
    upper_bound = safe_float(forecast_row.get("upper_bound"))

    spread = max(1.0, upper_bound - lower_bound)
    z_like = max(0.0, (predicted_incidents - lower_bound) / spread)

    return clamp(z_like, 0.0, 1.0)


def normalize_anomaly_component(anomaly_row: dict[str, Any] | None) -> float:
    if not anomaly_row:
        return 0.0

    is_anomaly = bool(anomaly_row.get("anomaly"))
    severity = str(anomaly_row.get("severity") or "normal").lower()

    if not is_anomaly or severity == "normal":
        return 0.0

    if severity == "high":
        return 1.0
    if severity == "medium":
        return 0.6
    if severity == "low":
        return 0.25

    return 0.0


def compute_risk_score(
    feature: dict[str, Any],
    baseline_stats: dict[str, dict[str, float]],
    forecast_row: dict[str, Any] | None,
    anomaly_row: dict[str, Any] | None,
) -> float:
    incidents_last_1h = max(0.0, safe_float(feature.get("incidents_last_1h")))
    incidents_last_3h = max(0.0, safe_float(feature.get("incidents_last_3h")))
    incidents_last_6h = max(0.0, safe_float(feature.get("incidents_last_6h")))
    incidents_last_24h = max(0.0, safe_float(feature.get("incidents_last_24h")))
    incidents_last_7d = max(0.0, safe_float(feature.get("incidents_last_7d")))

    open_ratio = clamp(safe_float(feature.get("open_active_ratio_24h")), 0.0, 1.0)
    online_ratio = clamp(safe_float(feature.get("filed_online_ratio_24h")), 0.0, 1.0)
    avg_delay = max(0.0, safe_float(feature.get("avg_report_delay_minutes_24h")))

    component_1h = normalize_metric_against_history(
        incidents_last_1h,
        baseline_stats.get("incidents_last_1h", {}),
    )
    component_3h = normalize_metric_against_history(
        incidents_last_3h,
        baseline_stats.get("incidents_last_3h", {}),
    )
    component_6h = normalize_metric_against_history(
        incidents_last_6h,
        baseline_stats.get("incidents_last_6h", {}),
    )
    component_24h = normalize_metric_against_history(
        incidents_last_24h,
        baseline_stats.get("incidents_last_24h", {}),
    )
    component_7d = normalize_metric_against_history(
        incidents_last_7d,
        baseline_stats.get("incidents_last_7d", {}),
    )
    delay_component = normalize_metric_against_history(
        avg_delay,
        baseline_stats.get("avg_report_delay_minutes_24h", {}),
    )

    forecast_component = normalize_forecast_component(forecast_row)
    anomaly_component = normalize_anomaly_component(anomaly_row)

    raw_score = (
        0.14 * component_1h
        + 0.14 * component_3h
        + 0.16 * component_6h
        + 0.18 * component_24h
        + 0.12 * component_7d
        + 0.10 * open_ratio
        + 0.03 * online_ratio
        + 0.04 * delay_component
        + 0.03 * forecast_component
        + 0.06 * anomaly_component
    )

    return round(clamp(raw_score * 100.0, 0.0, 100.0), 4)


def map_risk_level(score: float) -> str:
    if score >= 85:
        return "Very High"
    if score >= 65:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def build_risk_predictions(generated_at: datetime, target_timestamp: datetime) -> None:
    history_weeks = int(os.environ.get("RISK_HISTORY_WEEKS", str(DEFAULT_RISK_HISTORY_WEEKS)))
    min_history_points = int(os.environ.get("RISK_MIN_HISTORY_POINTS", str(DEFAULT_RISK_MIN_HISTORY_POINTS)))

    feature_rows = fetch_recent_risk_features(target_timestamp)
    strict_rows = fetch_risk_history_strict(target_timestamp, history_weeks)
    fallback_rows = fetch_risk_history_fallback(target_timestamp, history_weeks)
    forecast_map = fetch_forecast_map_for_risk(target_timestamp)
    anomaly_map = fetch_anomaly_map_for_risk(target_timestamp)

    strict_groupings = build_risk_groupings(strict_rows)
    fallback_groupings = build_risk_groupings(fallback_rows)

    results: list[dict[str, Any]] = []
    baseline_modes: dict[str, int] = {}

    for row in feature_rows:
        district = row.get("police_district")
        category = row.get("incident_category")
        if not district or not category:
            continue

        key = (district, category)
        baseline_stats, baseline_mode = select_risk_baseline_stats(
            district=district,
            category=category,
            strict_groupings=strict_groupings,
            fallback_groupings=fallback_groupings,
            min_history_points=min_history_points,
        )
        baseline_modes[baseline_mode] = baseline_modes.get(baseline_mode, 0) + 1

        if not baseline_stats:
            continue

        score = compute_risk_score(
            feature=row,
            baseline_stats=baseline_stats,
            forecast_row=forecast_map.get(key),
            anomaly_row=anomaly_map.get(key),
        )

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
    print(f"Risk baseline modes: {baseline_modes}")


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
    print(f"risk_model_name={RISK_MODEL_NAME}")

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
