import json
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import execute_etl_query

LOOKBACK_INTERVAL = os.environ.get("RISK_FEATURES_LOOKBACK_INTERVAL", os.environ.get("HISTORY_LOOKBACK_INTERVAL", "6 months"))
END_INTERVAL = os.environ.get("RISK_FEATURES_END_INTERVAL", "0 hours")
WARMUP_INTERVAL = os.environ.get("RISK_FEATURES_WARMUP_INTERVAL", "9 days")


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


def build_category_condition(alias: str, indentation: str = "  ") -> str:
    categories = load_category_filter()
    if not categories:
        return ""
    values = ", ".join(sql_quote(category) for category in categories)
    return f"\n{indentation}AND COALESCE({alias}.incident_category, 'Unknown') IN ({values})"


FEATURE_CATEGORY_CONDITION = build_category_condition("h", "    ")
DELAY_CATEGORY_CONDITION = build_category_condition("r", "      ")
HOURLY_CATEGORY_CONDITION = build_category_condition("ich", "   ")

SQL = f"""
BEGIN;

WITH params AS (
    SELECT
        date_trunc('hour', NOW() - INTERVAL {sql_interval(LOOKBACK_INTERVAL)}) AS start_ts,
        date_trunc('hour', NOW() - INTERVAL {sql_interval(END_INTERVAL)}) AS end_ts
)
DELETE FROM risk_features_hourly rf
USING params p
WHERE rf.feature_timestamp >= p.start_ts
  AND rf.feature_timestamp <  p.end_ts + INTERVAL '1 hour';

WITH params AS (
    SELECT
        date_trunc('hour', NOW() - INTERVAL {sql_interval(LOOKBACK_INTERVAL)}) AS start_ts,
        date_trunc('hour', NOW() - INTERVAL {sql_interval(END_INTERVAL)}) AS end_ts,
        INTERVAL {sql_interval(WARMUP_INTERVAL)} AS warmup_interval
),
feature_hours AS (
    SELECT DISTINCT
        h.bucket_start AS feature_timestamp,
        h.police_district,
        h.incident_category
    FROM incident_counts_hourly h
    CROSS JOIN params p
    WHERE h.bucket_start >= p.start_ts
      AND h.bucket_start <  p.end_ts + INTERVAL '1 hour'{FEATURE_CATEGORY_CONDITION}
),
delay_hourly AS (
    SELECT
        date_trunc('hour', r.incident_datetime) AS bucket_start,
        COALESCE(r.police_district, 'Unknown') AS police_district,
        COALESCE(r.incident_category, 'Unknown') AS incident_category,
        SUM(r.report_delay_minutes) AS sum_report_delay_minutes,
        COUNT(r.report_delay_minutes) AS count_report_delay_minutes
    FROM incidents_raw r
    CROSS JOIN params p
    WHERE r.incident_datetime IS NOT NULL
      AND r.incident_datetime >= p.start_ts - p.warmup_interval
      AND r.incident_datetime <  p.end_ts + INTERVAL '1 hour'{DELAY_CATEGORY_CONDITION}
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
    print(f"Rebuilding risk_features_hourly with lookback interval: {LOOKBACK_INTERVAL}")
    execute_etl_query(SQL)
