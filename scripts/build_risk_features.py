from utils import execute_query

SQL = """
TRUNCATE TABLE risk_features_hourly;

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
    date_trunc('hour', i.incident_datetime) AS feature_timestamp,
    COALESCE(i.police_district, 'Unknown') AS police_district,
    COALESCE(i.incident_category, 'Unknown') AS incident_category,
    EXTRACT(HOUR FROM i.incident_datetime)::int AS hour_of_day,
    TO_CHAR(i.incident_datetime, 'FMDay') AS day_of_week,
    EXTRACT(MONTH FROM i.incident_datetime)::int AS month_of_year,

    COUNT(*) FILTER (
        WHERE i2.incident_datetime >= date_trunc('hour', i.incident_datetime)
          AND i2.incident_datetime <  date_trunc('hour', i.incident_datetime) + interval '1 hour'
          AND i2.police_district = i.police_district
          AND i2.incident_category = i.incident_category
    ) AS incidents_last_1h,

    COUNT(*) FILTER (
        WHERE i2.incident_datetime >= date_trunc('hour', i.incident_datetime) - interval '2 hour'
          AND i2.incident_datetime <  date_trunc('hour', i.incident_datetime) + interval '1 hour'
          AND i2.police_district = i.police_district
          AND i2.incident_category = i.incident_category
    ) AS incidents_last_3h,

    COUNT(*) FILTER (
        WHERE i2.incident_datetime >= date_trunc('hour', i.incident_datetime) - interval '5 hour'
          AND i2.incident_datetime <  date_trunc('hour', i.incident_datetime) + interval '1 hour'
          AND i2.police_district = i.police_district
          AND i2.incident_category = i.incident_category
    ) AS incidents_last_6h,

    COUNT(*) FILTER (
        WHERE i2.incident_datetime >= date_trunc('hour', i.incident_datetime) - interval '23 hour'
          AND i2.incident_datetime <  date_trunc('hour', i.incident_datetime) + interval '1 hour'
          AND i2.police_district = i.police_district
          AND i2.incident_category = i.incident_category
    ) AS incidents_last_24h,

    COUNT(*) FILTER (
        WHERE i2.incident_datetime >= date_trunc('hour', i.incident_datetime) - interval '6 day'
          AND i2.incident_datetime <  date_trunc('hour', i.incident_datetime) + interval '1 hour'
          AND i2.police_district = i.police_district
          AND i2.incident_category = i.incident_category
    ) AS incidents_last_7d,

    AVG(CASE WHEN i2.resolution = 'Open or Active' THEN 1.0 ELSE 0.0 END) FILTER (
        WHERE i2.incident_datetime >= date_trunc('hour', i.incident_datetime) - interval '23 hour'
          AND i2.incident_datetime <  date_trunc('hour', i.incident_datetime) + interval '1 hour'
          AND i2.police_district = i.police_district
          AND i2.incident_category = i.incident_category
    ) AS open_active_ratio_24h,

    AVG(CASE WHEN i2.filed_online = true THEN 1.0 ELSE 0.0 END) FILTER (
        WHERE i2.incident_datetime >= date_trunc('hour', i.incident_datetime) - interval '23 hour'
          AND i2.incident_datetime <  date_trunc('hour', i.incident_datetime) + interval '1 hour'
          AND i2.police_district = i.police_district
          AND i2.incident_category = i.incident_category
    ) AS filed_online_ratio_24h,

    AVG(i2.report_delay_minutes) FILTER (
        WHERE i2.incident_datetime >= date_trunc('hour', i.incident_datetime) - interval '23 hour'
          AND i2.incident_datetime <  date_trunc('hour', i.incident_datetime) + interval '1 hour'
          AND i2.police_district = i.police_district
          AND i2.incident_category = i.incident_category
    ) AS avg_report_delay_minutes_24h
FROM incidents_raw i
JOIN incidents_raw i2
  ON i2.incident_datetime IS NOT NULL
WHERE i.incident_datetime IS NOT NULL
GROUP BY 1,2,3,4,5,6;
"""

if __name__ == "__main__":
    execute_query(SQL)