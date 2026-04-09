from utils import execute_query

SQL = """
TRUNCATE TABLE incident_counts_hourly;

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
    date_trunc('hour', incident_datetime) AS bucket_start,
    COALESCE(police_district, 'Unknown') AS police_district,
    COALESCE(incident_category, 'Unknown') AS incident_category,
    COALESCE(incident_subcategory, 'Unknown') AS incident_subcategory,
    COUNT(*) AS total_incidents,
    COUNT(*) FILTER (WHERE resolution = 'Open or Active') AS open_active_count,
    COUNT(*) FILTER (WHERE filed_online = true) AS filed_online_count
FROM incidents_raw
WHERE incident_datetime IS NOT NULL
GROUP BY 1,2,3,4;
"""

if __name__ == "__main__":
    execute_query(SQL)