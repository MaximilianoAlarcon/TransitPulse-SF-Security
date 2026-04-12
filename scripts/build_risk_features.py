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
    return f"\n      AND COALESCE({alias}.incident_category, 'Unknown') IN ({values})"


HOURLY_CATEGORY_CONDITION = build_category_condition("ich")
DELAY_CATEGORY_CONDITION = build_category_condition("r")
FEATURE_CATEGORY_CONDITION = build_category_condition("h")

SQL = f"""
BEGIN;

DELETE FROM risk_features_hourly
WHERE feature_timestamp >= date_trunc('hour', NOW() - INTERVAL '48 hours');

WITH feature_hours AS (
    SELECT DISTINCT
        bucket_start AS feature_timestamp,
        police_district,
        incident_category
    FROM incident_counts_hourly h
    WHERE bucket_start >= date_trunc('hour', NOW() - INTERVAL '48 hours'){FEATURE_CATEGORY_CONDITION}
),
delay_hourly AS (
    SELECT
        date_trunc('hour', incident_datetime) AS bucket_start,
        COALESCE(police_district, 'Unknown') AS police_district,
        COALESCE(incident_category, 'Unknown') AS incident_category,
        SUM(report_delay_minutes) AS sum_report_delay_minutes,
        COUNT(report_delay_minutes) AS count_report_delay_minutes
    FROM incidents_raw r
    WHERE incident_datetime IS NOT NULL
      AND incident_datetime >= date_trunc('hour', NOW() - INTERVAL '9 days'){DELAY_CATEGORY_CONDITION}
    GROUP BY 1,2,3
)

INSERT INTO risk_features_hourly (
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
)
SELECT
    fh.feature_timestamp,
    fh.police_district,
    fh.incident_category,
    EXTRACT(HOUR FROM fh.feature_timestamp)::int AS hour_of_day,
    TO_CHAR(fh.feature_timestamp, 'FMDay') AS day_of_week,
    EXTRACT(MONTH FROM fh.feature_timestamp)::int AS month_of_year,

    COALESCE(SUM(ich.total_incidents) FILTER (
        WHERE ich.bucket_start >= fh.feature_timestamp
          AND ich.bucket_start <  fh.feature_timestamp + INTERVAL '1 hour'
    ), 0) AS incidents_last_1h,

    COALESCE(SUM(ich.total_incidents) FILTER (
        WHERE ich.bucket_start >= fh.feature_timestamp - INTERVAL '2 hours'
          AND ich.bucket_start <  fh.feature_timestamp + INTERVAL '1 hour'
    ), 0) AS incidents_last_3h,

    COALESCE(SUM(ich.total_incidents) FILTER (
        WHERE ich.bucket_start >= fh.feature_timestamp - INTERVAL '5 hours'
          AND ich.bucket_start <  fh.feature_timestamp + INTERVAL '1 hour'
    ), 0) AS incidents_last_6h,

    COALESCE(SUM(ich.total_incidents) FILTER (
        WHERE ich.bucket_start >= fh.feature_timestamp - INTERVAL '23 hours'
          AND ich.bucket_start <  fh.feature_timestamp + INTERVAL '1 hour'
    ), 0) AS incidents_last_24h,

    COALESCE(SUM(ich.total_incidents) FILTER (
        WHERE ich.bucket_start >= fh.feature_timestamp - INTERVAL '6 days'
          AND ich.bucket_start <  fh.feature_timestamp + INTERVAL '1 hour'
    ), 0) AS incidents_last_7d,

    CASE
        WHEN COALESCE(SUM(ich.total_incidents) FILTER (
            WHERE ich.bucket_start >= fh.feature_timestamp - INTERVAL '23 hours'
              AND ich.bucket_start <  fh.feature_timestamp + INTERVAL '1 hour'
        ), 0) = 0
        THEN NULL
        ELSE
            SUM(ich.open_active_count) FILTER (
                WHERE ich.bucket_start >= fh.feature_timestamp - INTERVAL '23 hours'
                  AND ich.bucket_start <  fh.feature_timestamp + INTERVAL '1 hour'
            )::double precision
            /
            SUM(ich.total_incidents) FILTER (
                WHERE ich.bucket_start >= fh.feature_timestamp - INTERVAL '23 hours'
                  AND ich.bucket_start <  fh.feature_timestamp + INTERVAL '1 hour'
            )::double precision
    END AS open_active_ratio_24h,

    CASE
        WHEN COALESCE(SUM(ich.total_incidents) FILTER (
            WHERE ich.bucket_start >= fh.feature_timestamp - INTERVAL '23 hours'
              AND ich.bucket_start <  fh.feature_timestamp + INTERVAL '1 hour'
        ), 0) = 0
        THEN NULL
        ELSE
            SUM(ich.filed_online_count) FILTER (
                WHERE ich.bucket_start >= fh.feature_timestamp - INTERVAL '23 hours'
                  AND ich.bucket_start <  fh.feature_timestamp + INTERVAL '1 hour'
            )::double precision
            /
            SUM(ich.total_incidents) FILTER (
                WHERE ich.bucket_start >= fh.feature_timestamp - INTERVAL '23 hours'
                  AND ich.bucket_start <  fh.feature_timestamp + INTERVAL '1 hour'
            )::double precision
    END AS filed_online_ratio_24h,

    CASE
        WHEN COALESCE(SUM(dh.count_report_delay_minutes) FILTER (
            WHERE dh.bucket_start >= fh.feature_timestamp - INTERVAL '23 hours'
              AND dh.bucket_start <  fh.feature_timestamp + INTERVAL '1 hour'
        ), 0) = 0
        THEN NULL
        ELSE
            SUM(dh.sum_report_delay_minutes) FILTER (
                WHERE dh.bucket_start >= fh.feature_timestamp - INTERVAL '23 hours'
                  AND dh.bucket_start <  fh.feature_timestamp + INTERVAL '1 hour'
            )::double precision
            /
            SUM(dh.count_report_delay_minutes) FILTER (
                WHERE dh.bucket_start >= fh.feature_timestamp - INTERVAL '23 hours'
                  AND dh.bucket_start <  fh.feature_timestamp + INTERVAL '1 hour'
            )::double precision
    END AS avg_report_delay_minutes_24h

FROM feature_hours fh
LEFT JOIN incident_counts_hourly ich
    ON ich.police_district = fh.police_district
   AND ich.incident_category = fh.incident_category
   AND ich.bucket_start >= fh.feature_timestamp - INTERVAL '6 days'
   AND ich.bucket_start <  fh.feature_timestamp + INTERVAL '1 hour'{HOURLY_CATEGORY_CONDITION}
LEFT JOIN delay_hourly dh
    ON dh.police_district = fh.police_district
   AND dh.incident_category = fh.incident_category
   AND dh.bucket_start >= fh.feature_timestamp - INTERVAL '23 hours'
   AND dh.bucket_start <  fh.feature_timestamp + INTERVAL '1 hour'
GROUP BY
    fh.feature_timestamp,
    fh.police_district,
    fh.incident_category;

COMMIT;
"""

if __name__ == "__main__":
    execute_etl_query(SQL)
