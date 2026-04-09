from utils import execute_etl_query

SQL = """
TRUNCATE TABLE incident_counts_daily;

INSERT INTO incident_counts_daily (
    bucket_date,
    police_district,
    incident_category,
    incident_subcategory,
    total_incidents,
    open_active_count,
    filed_online_count
)
SELECT
    incident_date AS bucket_date,
    COALESCE(police_district, 'Unknown'),
    COALESCE(incident_category, 'Unknown'),
    COALESCE(incident_subcategory, 'Unknown'),
    COUNT(*),
    COUNT(*) FILTER (WHERE resolution = 'Open or Active'),
    COUNT(*) FILTER (WHERE filed_online = true)
FROM incidents_raw
WHERE incident_date IS NOT NULL
GROUP BY 1,2,3,4;
"""

if __name__ == "__main__":
    execute_etl_query(SQL)