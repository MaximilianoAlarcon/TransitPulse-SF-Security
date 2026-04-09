import requests
from utils import get_connection

API_URL = "https://data.sfgov.org/resource/wg3w-h783.json?$limit=1000"

def load_incidents():
    response = requests.get(API_URL, timeout=60)
    response.raise_for_status()
    rows = response.json()

    insert_sql = """
    INSERT INTO incidents_raw (
        row_id,
        incident_datetime,
        incident_date,
        incident_time,
        incident_year,
        incident_day_of_week,
        report_datetime,
        incident_id,
        incident_number,
        report_type_code,
        report_type_description,
        filed_online,
        incident_code,
        incident_category,
        incident_subcategory,
        incident_description,
        resolution,
        police_district,
        data_as_of,
        data_loaded_at,
        incident_hour,
        report_delay_minutes
    )
    VALUES (
        %(row_id)s,
        %(incident_datetime)s,
        %(incident_date)s,
        %(incident_time)s,
        %(incident_year)s,
        %(incident_day_of_week)s,
        %(report_datetime)s,
        %(incident_id)s,
        %(incident_number)s,
        %(report_type_code)s,
        %(report_type_description)s,
        %(filed_online)s,
        %(incident_code)s,
        %(incident_category)s,
        %(incident_subcategory)s,
        %(incident_description)s,
        %(resolution)s,
        %(police_district)s,
        %(data_as_of)s,
        %(data_loaded_at)s,
        EXTRACT(HOUR FROM %(incident_datetime)s::timestamp),
        CASE
            WHEN %(report_datetime)s IS NOT NULL AND %(incident_datetime)s IS NOT NULL
            THEN EXTRACT(EPOCH FROM (%(report_datetime)s::timestamp - %(incident_datetime)s::timestamp)) / 60
            ELSE NULL
        END
    )
    ON CONFLICT (row_id) DO UPDATE SET
        incident_datetime = EXCLUDED.incident_datetime,
        incident_date = EXCLUDED.incident_date,
        incident_time = EXCLUDED.incident_time,
        incident_year = EXCLUDED.incident_year,
        incident_day_of_week = EXCLUDED.incident_day_of_week,
        report_datetime = EXCLUDED.report_datetime,
        incident_id = EXCLUDED.incident_id,
        incident_number = EXCLUDED.incident_number,
        report_type_code = EXCLUDED.report_type_code,
        report_type_description = EXCLUDED.report_type_description,
        filed_online = EXCLUDED.filed_online,
        incident_code = EXCLUDED.incident_code,
        incident_category = EXCLUDED.incident_category,
        incident_subcategory = EXCLUDED.incident_subcategory,
        incident_description = EXCLUDED.incident_description,
        resolution = EXCLUDED.resolution,
        police_district = EXCLUDED.police_district,
        data_as_of = EXCLUDED.data_as_of,
        data_loaded_at = EXCLUDED.data_loaded_at,
        incident_hour = EXCLUDED.incident_hour,
        report_delay_minutes = EXCLUDED.report_delay_minutes;
    """

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                for row in rows:
                    cur.execute(insert_sql, row)
    finally:
        conn.close()

if __name__ == "__main__":
    load_incidents()