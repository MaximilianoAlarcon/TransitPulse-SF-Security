import os
import sys
from datetime import datetime, timedelta, timezone

import requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import get_db_connection

BASE_API_URL = "https://data.sfgov.org/resource/wg3w-h783.json"
PAGE_SIZE = 1000
OVERLAP_HOURS = 2

EXPECTED_FIELDS = [
    "row_id",
    "incident_datetime",
    "incident_date",
    "incident_time",
    "incident_year",
    "incident_day_of_week",
    "report_datetime",
    "incident_id",
    "incident_number",
    "report_type_code",
    "report_type_description",
    "filed_online",
    "incident_code",
    "incident_category",
    "incident_subcategory",
    "incident_description",
    "resolution",
    "police_district",
    "data_as_of",
    "data_loaded_at",
]


def normalize_row(row: dict) -> dict:
    return {field: row.get(field) for field in EXPECTED_FIELDS}


def get_last_incident_datetime() -> datetime | None:
    query = "SELECT MAX(incident_datetime) AS max_incident_datetime FROM incidents_raw;"

    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(query)
                result = cur.fetchone()

                if not result:
                    return None

                # Compatible con cursor tipo dict o tuple
                if isinstance(result, dict):
                    value = result.get("max_incident_datetime")
                else:
                    value = result[0]

                return value
    finally:
        conn.close()


def format_socrata_datetime(dt: datetime) -> str:
    """
    Socrata acepta bien formato ISO tipo:
    2026-02-21T10:00:00
    """
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def fetch_rows_since(last_dt: datetime | None) -> list[dict]:
    all_rows: list[dict] = []
    offset = 0

    params = {
        "$limit": PAGE_SIZE,
        "$offset": offset,
        "$order": "incident_datetime ASC",
    }

    if last_dt is not None:
        safe_dt = last_dt - timedelta(hours=OVERLAP_HOURS)
        params["$where"] = f"incident_datetime >= '{format_socrata_datetime(safe_dt)}'"

    while True:
        params["$offset"] = offset

        response = requests.get(BASE_API_URL, params=params, timeout=60)
        response.raise_for_status()

        page_rows = response.json()
        if not page_rows:
            break

        all_rows.extend(page_rows)

        print(
            f"Fetched {len(page_rows)} rows "
            f"(offset={offset}, total_so_far={len(all_rows)})"
        )

        if len(page_rows) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

    return all_rows


def upsert_rows(rows: list[dict]) -> None:
    if not rows:
        print("No new rows to upsert.")
        return

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
        CASE
            WHEN %(incident_datetime)s IS NOT NULL
            THEN EXTRACT(HOUR FROM %(incident_datetime)s::timestamp)
            ELSE NULL
        END,
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

    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                for row in rows:
                    cur.execute(insert_sql, normalize_row(row))
    finally:
        conn.close()

    print(f"Upsert completed for {len(rows)} rows.")


def load_incidents() -> None:
    last_dt = get_last_incident_datetime()
    print(f"Last incident_datetime in DB: {last_dt}")

    rows = fetch_rows_since(last_dt)
    print(f"Total fetched from API: {len(rows)}")

    upsert_rows(rows)


if __name__ == "__main__":
    load_incidents()