import json
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import execute_etl_query

LOOKBACK_INTERVAL = os.environ.get("HOURLY_AGG_LOOKBACK_INTERVAL", os.environ.get("HISTORY_LOOKBACK_INTERVAL", "6 months"))
END_INTERVAL = os.environ.get("HOURLY_AGG_END_INTERVAL", "0 hours")


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


CATEGORY_CONDITION = build_category_condition("r")

SQL = f"""
BEGIN;

WITH params AS (
    SELECT
        date_trunc('hour', NOW() - INTERVAL {sql_interval(LOOKBACK_INTERVAL)}) AS start_ts,
        date_trunc('hour', NOW() - INTERVAL {sql_interval(END_INTERVAL)}) AS end_ts
)
DELETE FROM incident_counts_hourly h
USING params p
WHERE h.bucket_start >= p.start_ts
  AND h.bucket_start <  p.end_ts + INTERVAL '1 hour';

WITH params AS (
    SELECT
        date_trunc('hour', NOW() - INTERVAL {sql_interval(LOOKBACK_INTERVAL)}) AS start_ts,
        date_trunc('hour', NOW() - INTERVAL {sql_interval(END_INTERVAL)}) AS end_ts
)
INSERT INTO incident_counts_hourly (
    bucket_start,
    police_district,
    incident_category,
    incident_subcategory,
    total_incidents,
    open_active_count,
    filed_online_count
)
SELECT
    date_trunc('hour', r.incident_datetime) AS bucket_start,
    COALESCE(r.police_district, 'Unknown') AS police_district,
    COALESCE(r.incident_category, 'Unknown') AS incident_category,
    COALESCE(r.incident_subcategory, 'Unknown') AS incident_subcategory,
    COUNT(*) AS total_incidents,
    COUNT(*) FILTER (WHERE r.resolution = 'Open or Active') AS open_active_count,
    COUNT(*) FILTER (WHERE r.filed_online = true) AS filed_online_count
FROM incidents_raw r
CROSS JOIN params p
WHERE r.incident_datetime IS NOT NULL
  AND r.incident_datetime >= p.start_ts
  AND r.incident_datetime <  p.end_ts + INTERVAL '1 hour'{CATEGORY_CONDITION}
GROUP BY 1,2,3,4;

COMMIT;
"""

if __name__ == "__main__":
    print(f"Refreshing incident_counts_hourly with lookback interval: {LOOKBACK_INTERVAL}")
    execute_etl_query(SQL)
