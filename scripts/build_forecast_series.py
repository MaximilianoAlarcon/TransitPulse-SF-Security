import json
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import execute_etl_query

LOOKBACK_INTERVAL = os.environ.get("FORECAST_SERIES_LOOKBACK_INTERVAL", os.environ.get("HISTORY_LOOKBACK_INTERVAL", "6 months"))
END_INTERVAL = os.environ.get("FORECAST_SERIES_END_INTERVAL", "0 hours")


def load_category_filter() -> list[str]:
    raw = os.environ.get("CATEGORY_FILTER_VALUES_JSON", "[]")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = []
    return [str(item).strip() for item in parsed if str(item).strip()]


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sql_interval(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_category_condition(alias: str) -> str:
    categories = load_category_filter()
    if not categories:
        return ""
    values = ", ".join(sql_quote(category) for category in categories)
    return f"\n  AND COALESCE({alias}.incident_category, 'Unknown') IN ({values})"


CATEGORY_CONDITION = build_category_condition("h")

SQL = f"""
BEGIN;

WITH params AS (
    SELECT
        date_trunc('hour', NOW() - INTERVAL {sql_interval(LOOKBACK_INTERVAL)}) AS start_ts,
        date_trunc('hour', NOW() - INTERVAL {sql_interval(END_INTERVAL)}) AS end_ts
)
DELETE FROM forecast_training_series f
USING params p
WHERE f.bucket_start >= p.start_ts
  AND f.bucket_start <  p.end_ts + INTERVAL '1 hour';

WITH params AS (
    SELECT
        date_trunc('hour', NOW() - INTERVAL {sql_interval(LOOKBACK_INTERVAL)}) AS start_ts,
        date_trunc('hour', NOW() - INTERVAL {sql_interval(END_INTERVAL)}) AS end_ts
)
INSERT INTO forecast_training_series (
    series_id,
    bucket_start,
    police_district,
    incident_category,
    total_incidents
)
SELECT
    CONCAT(h.police_district, '|', h.incident_category) AS series_id,
    h.bucket_start,
    h.police_district,
    h.incident_category,
    SUM(h.total_incidents) AS total_incidents
FROM incident_counts_hourly h
CROSS JOIN params p
WHERE h.bucket_start >= p.start_ts
  AND h.bucket_start <  p.end_ts + INTERVAL '1 hour'{CATEGORY_CONDITION}
GROUP BY 1,2,3,4;

COMMIT;
"""

if __name__ == "__main__":
    print(f"Rebuilding forecast_training_series with lookback interval: {LOOKBACK_INTERVAL}")
    execute_etl_query(SQL)
