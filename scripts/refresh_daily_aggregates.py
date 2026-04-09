import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import execute_etl_query

SQL = """
BEGIN;

DELETE FROM incident_counts_daily
WHERE bucket_date >= CURRENT_DATE - INTERVAL '7 days';

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
    COALESCE(police_district, 'Unknown') AS police_district,
    COALESCE(incident_category, 'Unknown') AS incident_category,
    COALESCE(incident_subcategory, 'Unknown') AS incident_subcategory,
    COUNT(*) AS total_incidents,
    COUNT(*) FILTER (WHERE resolution = 'Open or Active') AS open_active_count,
    COUNT(*) FILTER (WHERE filed_online = true) AS filed_online_count
FROM incidents_raw
WHERE incident_date IS NOT NULL
  AND incident_date >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY 1,2,3,4;

COMMIT;
"""

if __name__ == "__main__":
    execute_etl_query(SQL)