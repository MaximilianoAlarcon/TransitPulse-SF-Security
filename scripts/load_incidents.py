import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any

import requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import get_db_connection

BASE_API_URL = "https://data.sfgov.org/resource/wg3w-h783.json"
PAGE_SIZE = 1000
OVERLAP_HOURS = 2
BACKFILL_BATCH_SIZE = 200

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
    "latitude",
    "longitude",
]


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_coordinates(row: dict[str, Any]) -> tuple[float | None, float | None]:
    lat = safe_float(row.get("latitude"))
    lon = safe_float(row.get("longitude"))

    if lat is not None and lon is not None:
        return lat, lon

    for key in ("point", "location", "geolocation", "intersection"):
        raw = row.get(key)
        if not raw:
            continue

        if isinstance(raw, dict):
            lat = safe_float(raw.get("latitude"))
            lon = safe_float(raw.get("longitude"))
            if lat is not None and lon is not None:
                return lat, lon

            coords = raw.get("coordinates")
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                lon = safe_float(coords[0])
                lat = safe_float(coords[1])
                if lat is not None and lon is not None:
                    return lat, lon

    return None, None


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {field: row.get(field) for field in EXPECTED_FIELDS if field not in {"latitude", "longitude"}}
    lat, lon = extract_coordinates(row)
    normalized["latitude"] = lat
    normalized["longitude"] = lon
    return normalized


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

                if isinstance(result, dict):
                    return result.get("max_incident_datetime")
                return result[0]
    finally:
        conn.close()


def get_missing_coordinate_row_ids(limit: int = 5000) -> list[str]:
    query = """
    SELECT row_id
    FROM incidents_raw
    WHERE latitude IS NULL OR longitude IS NULL
    ORDER BY incident_datetime DESC NULLS LAST
    LIMIT %s;
    """

    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(query, (limit,))
                rows = cur.fetchall()
                values = []
                for row in rows:
                    if isinstance(row, dict):
                        values.append(row.get("row_id"))
                    else:
                        values.append(row[0])
                return [value for value in values if value]
    finally:
        conn.close()


def format_socrata_datetime(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def fetch_rows_since(last_dt: datetime | None) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
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


def fetch_rows_by_row_ids(row_ids: list[str]) -> list[dict[str, Any]]:
    if not row_ids:
        return []

    escaped_ids = ",".join("'" + row_id.replace("'", "''") + "'" for row_id in row_ids)
    params = {
        "$limit": len(row_ids),
        "$where": f"row_id IN ({escaped_ids})",
    }

    response = requests.get(BASE_API_URL, params=params, timeout=60)
    response.raise_for_status()
    rows = response.json()
    print(f"Backfill fetched {len(rows)} rows for {len(row_ids)} requested row_ids.")
    return rows


def upsert_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No rows to upsert.")
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
        report_delay_minutes,
        latitude,
        longitude,
        geom
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
        END,
        %(latitude)s,
        %(longitude)s,
        CASE
            WHEN %(latitude)s IS NOT NULL AND %(longitude)s IS NOT NULL
            THEN ST_SetSRID(ST_MakePoint(%(longitude)s, %(latitude)s), 4326)
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
        report_delay_minutes = EXCLUDED.report_delay_minutes,
        latitude = COALESCE(EXCLUDED.latitude, incidents_raw.latitude),
        longitude = COALESCE(EXCLUDED.longitude, incidents_raw.longitude),
        geom = COALESCE(EXCLUDED.geom, incidents_raw.geom),
        updated_at = NOW();
    """

    normalized_rows = [normalize_row(row) for row in rows]

    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                for row in normalized_rows:
                    cur.execute(insert_sql, row)
    finally:
        conn.close()

    with_coords = sum(1 for row in normalized_rows if row.get("latitude") is not None and row.get("longitude") is not None)
    print(f"Upsert completed for {len(normalized_rows)} rows ({with_coords} with coordinates).")


def backfill_missing_coordinates(limit: int = 5000) -> None:
    missing_row_ids = get_missing_coordinate_row_ids(limit=limit)
    if not missing_row_ids:
        print("No rows with missing coordinates found.")
        return

    print(f"Found {len(missing_row_ids)} rows with missing coordinates to backfill.")

    total_processed = 0
    for start in range(0, len(missing_row_ids), BACKFILL_BATCH_SIZE):
        batch_ids = missing_row_ids[start:start + BACKFILL_BATCH_SIZE]
        rows = fetch_rows_by_row_ids(batch_ids)
        upsert_rows(rows)
        total_processed += len(batch_ids)

    print(f"Backfill completed for {total_processed} requested row_ids.")


def load_incidents() -> None:
    last_dt = get_last_incident_datetime()
    print(f"Last incident_datetime in DB: {last_dt}")

    rows = fetch_rows_since(last_dt)
    print(f"Total fetched from API: {len(rows)}")

    upsert_rows(rows)

    backfill_limit = int(os.environ.get("BACKFILL_MISSING_COORDS_LIMIT", "0"))
    if backfill_limit > 0:
        backfill_missing_coordinates(limit=backfill_limit)


if __name__ == "__main__":
    load_incidents()
