import json
import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import execute_etl_query


def load_category_filter() -> list[str]:
    raw = os.environ.get("CATEGORY_FILTER_VALUES_JSON", "[]")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = []
    return [str(item).strip() for item in parsed if str(item).strip()]


def sql_quote(value: str) -> str:
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

DELETE FROM forecast_training_series
WHERE bucket_start >= date_trunc('hour', NOW() - INTERVAL '48 hours');

INSERT INTO forecast_training_series (
    series_id,
    bucket_start,
    police_district,
    incident_category,
    total_incidents
)
SELECT
    CONCAT(police_district, '|', incident_category) AS series_id,
    bucket_start,
    police_district,
    incident_category,
    SUM(total_incidents) AS total_incidents
FROM incident_counts_hourly h
WHERE bucket_start >= date_trunc('hour', NOW() - INTERVAL '48 hours'){CATEGORY_CONDITION}
GROUP BY 1,2,3,4;

COMMIT;
"""

if __name__ == "__main__":
    execute_etl_query(SQL)
