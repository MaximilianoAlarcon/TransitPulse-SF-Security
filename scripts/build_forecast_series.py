import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import execute_etl_query

SQL = """
TRUNCATE TABLE forecast_training_series;

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
FROM incident_counts_hourly
GROUP BY 1,2,3,4;
"""

if __name__ == "__main__":
    execute_etl_query(SQL)