import csv
import io
import json
import math
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
from typing import Any, Callable

from flask import Flask, Response, jsonify, render_template, request
from psycopg2.extras import RealDictCursor

from utils import execute_query, execute_sql_file, get_db_connection

from zoneinfo import ZoneInfo

from ml_utils import (
    MODEL_BUCKET_NAME,
    MODEL_DIR,
    RISK_CLASSIFIER_MODEL_NAME,
    VOLUME_MODEL_NAME,
    build_risk_forecast_geojson,
    build_volume_forecast_geojson,
    fetch_latest_volume_features,
    get_model_artifact_keys,
    get_risk_model_artifact_keys,
    get_risk_model_path,
    get_volume_model_path,
    predict_risk_from_rows,
    predict_volume_from_rows,
    read_risk_model_metrics,
    read_volume_model_metrics,
)

BASE_DIR = Path(__file__).resolve().parent

DB_CONFIG = {
    "host": os.environ.get("DB_HOST"),
    "database": os.environ.get("DB_NAME"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASSWORD"),
    "port": os.environ.get("DB_PORT"),
}

app = Flask(__name__, template_folder="templates", static_folder="static")

WINDOW_TO_DELTA = {
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
}

SF_CENTER = {"lat": 37.7749, "lon": -122.4194}
MAP_POINT_LIMIT = 600


CATEGORY_FILTER_VALUES = [
    "Larceny Theft",
    "Drug Offense",
    "Drug Violation",
    "Assault",
    "Malicious Mischief",
    "Burglary",
    "Motor Vehicle Theft",
    "Disorderly Conduct",
    "Fraud",
    "Robbery",
    "Offences Against The Family And Children",
    "Weapons Offense",
    "Weapons Carrying Etc",
    "Forgery And Counterfeiting",
    "Arson",
    "Stolen Property",
    "Vandalism",
    "Embezzlement",
    "Liquor Laws",
    "Prostitution",
    "Homicide",
    "Gambling",
    "Sex Offense",
]



ALLOWED_ADMIN_ETL_COMMANDS = {
    "build_risk_features": {
        "script": "scripts/build_risk_features.py",
        "allowed_env": {
            "RISK_FEATURES_FULL_GRID",
            "RISK_FEATURES_LOOKBACK_INTERVAL",
            "RISK_FEATURES_END_INTERVAL",
            "RISK_FEATURES_WARMUP_INTERVAL",
            "HISTORY_LOOKBACK_INTERVAL",
        },
        "default_env": {
            "RISK_FEATURES_FULL_GRID": "true",
            "RISK_FEATURES_LOOKBACK_INTERVAL": "6 months",
        },
    },
    "build_forecast_series": {
        "script": "scripts/build_forecast_series.py",
        "allowed_env": {
            "FORECAST_SERIES_LOOKBACK_INTERVAL",
            "HISTORY_LOOKBACK_INTERVAL",
        },
        "default_env": {},
    },
    "refresh_hourly_aggregates": {
        "script": "scripts/refresh_hourly_aggregates.py",
        "allowed_env": {
            "HOURLY_AGG_LOOKBACK_INTERVAL",
            "HISTORY_LOOKBACK_INTERVAL",
        },
        "default_env": {},
    },
    "build_predictions": {
        "script": "scripts/build_predictions.py",
        "allowed_env": {
            "RISK_MODEL_NAME",
            "RISK_HISTORY_WEEKS",
            "RISK_MIN_HISTORY_POINTS",
            "FORECAST_HISTORY_WEEKS",
            "FORECAST_MIN_HISTORY_POINTS",
        },
        "default_env": {},
    },
}

ADMIN_ETL_MAX_TIMEOUT_SECONDS = int(os.environ.get("ADMIN_ETL_MAX_TIMEOUT_SECONDS", "1800"))

def fetch_all_dict(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]


def fetch_one_dict(query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            row = cur.fetchone()
            return dict(row) if row else None


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def build_csv_response(filename: str, csv_text: str) -> Response:
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def build_zip_response(filename: str, files: dict[str, str]) -> Response:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for inner_name, content in files.items():
            zf.writestr(inner_name, content)

    buffer.seek(0)

    return Response(
        buffer.getvalue(),
        mimetype="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def safe_int(value: Any) -> int:
    return 0 if value is None else int(value)


def safe_float(value: Any, digits: int = 2) -> float:
    return 0.0 if value is None else round(float(value), digits)


def parse_filters() -> dict[str, Any]:
    window = request.args.get("window", "7d")
    district = (request.args.get("district") or "all").strip() or "all"
    category = (request.args.get("category") or "all").strip() or "all"

    if window not in WINDOW_TO_DELTA:
        window = "7d"

    end_dt = datetime.now(ZoneInfo("America/Los_Angeles"))
    start_dt = end_dt - WINDOW_TO_DELTA[window]

    return {
        "window": window,
        "district": district,
        "category": category,
        "start_dt": start_dt,
        "end_dt": end_dt,
    }


def apply_common_filters(alias: str, filters: dict[str, Any], timestamp_column: str) -> tuple[str, tuple[Any, ...]]:
    conditions = [
        f"{alias}.{timestamp_column} >= %s",
        f"{alias}.{timestamp_column} < %s",
    ]
    params: list[Any] = [filters["start_dt"], filters["end_dt"]]

    if filters["district"].lower() != "all":
        conditions.append(f"{alias}.police_district = %s")
        params.append(filters["district"])

    if filters["category"].lower() != "all":
        conditions.append(f"{alias}.incident_category = %s")
        params.append(filters["category"])

    return " AND ".join(conditions), tuple(params)


def is_open_resolution(value: Any) -> bool:
    text = (value or "").strip().lower()
    return text.startswith("open") or text.startswith("active")


def category_base_score(category: str) -> float:
    normalized = (category or "").strip().lower()
    high_keywords = ["assault", "robbery", "burglary", "weapon", "homicide", "arson", "sex offense"]
    medium_keywords = ["theft", "larceny", "fraud", "motor vehicle theft", "stolen property", "vandalism", "malicious mischief"]
    low_keywords = ["drug", "disorderly", "liquor", "gambling", "prostitution"]

    if any(keyword in normalized for keyword in high_keywords):
        return 0.9
    if any(keyword in normalized for keyword in medium_keywords):
        return 0.6
    if any(keyword in normalized for keyword in low_keywords):
        return 0.35
    return 0.3


def compute_point_risk(row: dict[str, Any], risk_mode: str) -> tuple[float, str]:
    base_score = category_base_score(row.get("incident_category") or "Unknown")
    resolution = row.get("resolution")
    delay_minutes = float(row.get("report_delay_minutes") or 0)

    if risk_mode == "open":
        score = 0.85 if is_open_resolution(resolution) else max(0.25, base_score * 0.55)
        label = "Open / Active"
    elif risk_mode == "delay":
        if delay_minutes >= 180:
            score = 0.9
        elif delay_minutes >= 90:
            score = 0.65
        elif delay_minutes >= 30:
            score = 0.45
        else:
            score = 0.25
        label = "Report delay"
    else:
        score = base_score
        label = "Volume"

    return max(0.0, min(1.0, score)), label


def require_admin_token(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        expected_token = os.environ.get("ADMIN_TOKEN")
        if not expected_token:
            return jsonify({"status": "error", "message": "ADMIN_TOKEN is not configured. Set it in Railway before using admin ML endpoints."}), 500

        header_token = request.headers.get("X-Admin-Token") or ""
        auth_header = request.headers.get("Authorization") or ""
        bearer_token = auth_header.removeprefix("Bearer ").strip() if auth_header.startswith("Bearer ") else ""
        provided_token = header_token.strip() or bearer_token

        if provided_token != expected_token:
            return jsonify({"status": "error", "message": "Unauthorized"}), 401

        return func(*args, **kwargs)

    return wrapper


def parse_admin_json_payload() -> dict[str, Any]:
    return request.get_json(silent=True) or {}


def parse_positive_int(value: Any, default: int, min_value: int = 1, max_value: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(parsed, max_value)
    return parsed


def parse_float_range(value: Any, default: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))



def build_category_filter_env() -> str:
    return json.dumps(CATEGORY_FILTER_VALUES)


def run_allowed_admin_etl(command_name: str, extra_env: dict[str, Any] | None, timeout_seconds: int) -> dict[str, Any]:
    if command_name not in ALLOWED_ADMIN_ETL_COMMANDS:
        allowed = sorted(ALLOWED_ADMIN_ETL_COMMANDS.keys())
        raise ValueError(f"Invalid command '{command_name}'. Allowed commands: {allowed}")

    command_config = ALLOWED_ADMIN_ETL_COMMANDS[command_name]
    script_path = BASE_DIR / command_config["script"]
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    allowed_env = command_config["allowed_env"]
    sanitized_env: dict[str, str] = {}

    for key, value in command_config.get("default_env", {}).items():
        sanitized_env[key] = str(value)

    for key, value in (extra_env or {}).items():
        if key not in allowed_env:
            raise ValueError(f"Environment variable '{key}' is not allowed for command '{command_name}'.")
        sanitized_env[key] = str(value)

    env = os.environ.copy()
    env.update(sanitized_env)
    env["CATEGORY_FILTER_VALUES_JSON"] = build_category_filter_env()

    safe_timeout = max(1, min(int(timeout_seconds), ADMIN_ETL_MAX_TIMEOUT_SECONDS))
    started_at = datetime.utcnow().replace(microsecond=0)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=safe_timeout,
    )

    finished_at = datetime.utcnow().replace(microsecond=0)
    stdout = result.stdout or ""
    stderr = result.stderr or ""

    response = {
        "command": command_name,
        "script": command_config["script"],
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "returncode": result.returncode,
        "env_overrides": sanitized_env,
        "stdout": stdout[-12000:],
        "stderr": stderr[-12000:],
    }

    if result.returncode != 0:
        raise RuntimeError(json.dumps(response, default=str))

    return response



@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/db-test")
def db_test():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_database(), version();")
                database_name, version = cur.fetchone()

        return jsonify(
            {
                "status": "ok",
                "database": database_name,
                "postgres_version": version,
                "postgis_expected": True,
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/init-db", methods=["POST"])
def init_db():
    sql_file = BASE_DIR / "db_structure.sql"
    if not sql_file.exists():
        return jsonify({"status": "error", "message": "db_structure.sql not found"}), 500

    try:
        execute_sql_file(DB_CONFIG, sql_file)
        return jsonify(
            {
                "status": "ok",
                "message": "Database structure created successfully",
                "sql_file": sql_file.name,
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/db-query", methods=["POST"])
def db_query():
    payload = request.get_json(silent=True) or {}
    query = payload.get("query", "")

    try:
        result = execute_query(DB_CONFIG, query)
        if result["has_result_set"]:
            return jsonify(
                {
                    "status": "ok",
                    "query": query,
                    "result_type": "result_set",
                    "columns": result["columns"],
                    "rows": result["rows"],
                    "row_count": result["row_count"],
                    "status_message": result["status_message"],
                }
            )

        return jsonify(
            {
                "status": "ok",
                "query": query,
                "result_type": "command",
                "affected_rows": result["affected_rows"],
                "status_message": result["status_message"],
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc), "query": query}), 500


@app.route("/api/debug/export-source-csv")
def api_debug_export_source_csv():
    table = (request.args.get("table") or "").strip().lower()
    limit = request.args.get("limit", type=int)
    export_all = (request.args.get("all") or "true").strip().lower() in {"true", "1", "yes"}

    allowed_tables = {
        "forecast_training_series": {
            "base_query": """
                SELECT bucket_start, police_district, incident_category, total_incidents
                FROM forecast_training_series
                ORDER BY bucket_start DESC
            """,
            "filename": "forecast_training_series.csv",
        },
        "incident_counts_hourly": {
            "base_query": """
                SELECT bucket_start, police_district, incident_category, incident_subcategory,
                       total_incidents, open_active_count, filed_online_count
                FROM incident_counts_hourly
                ORDER BY bucket_start DESC
            """,
            "filename": "incident_counts_hourly.csv",
        },
        "risk_features_hourly": {
            "base_query": """
                SELECT feature_timestamp, police_district, incident_category,
                       incidents_last_1h, incidents_last_3h, incidents_last_6h,
                       incidents_last_24h, incidents_last_7d,
                       open_active_ratio_24h, filed_online_ratio_24h,
                       avg_report_delay_minutes_24h
                FROM risk_features_hourly
                ORDER BY feature_timestamp DESC
            """,
            "filename": "risk_features_hourly.csv",
        },
    }

    if table not in allowed_tables:
        return jsonify(
            {
                "status": "error",
                "message": "Invalid table. Use forecast_training_series, incident_counts_hourly, or risk_features_hourly.",
            }
        ), 400

    config = allowed_tables[table]

    if export_all:
        rows = fetch_all_dict(config["base_query"])
    else:
        safe_limit = 200 if limit is None else max(1, min(limit, 50000))
        rows = fetch_all_dict(f"{config['base_query']} LIMIT %s;", (safe_limit,))

    csv_text = rows_to_csv(rows)
    return build_csv_response(config["filename"], csv_text)


@app.route("/api/debug/export-source-zip")
def api_debug_export_source_zip():
    limit = request.args.get("limit", type=int)
    export_all = (request.args.get("all") or "true").strip().lower() in {"true", "1", "yes"}

    table_configs = {
        "forecast_training_series.csv": """
            SELECT bucket_start, police_district, incident_category, total_incidents
            FROM forecast_training_series
            ORDER BY bucket_start DESC
        """,
        "incident_counts_hourly.csv": """
            SELECT bucket_start, police_district, incident_category, incident_subcategory,
                   total_incidents, open_active_count, filed_online_count
            FROM incident_counts_hourly
            ORDER BY bucket_start DESC
        """,
        "risk_features_hourly.csv": """
            SELECT feature_timestamp, police_district, incident_category,
                   incidents_last_1h, incidents_last_3h, incidents_last_6h,
                   incidents_last_24h, incidents_last_7d,
                   open_active_ratio_24h, filed_online_ratio_24h,
                   avg_report_delay_minutes_24h
            FROM risk_features_hourly
            ORDER BY feature_timestamp DESC
        """,
    }

    files: dict[str, str] = {}

    for filename, base_query in table_configs.items():
        if export_all:
            rows = fetch_all_dict(base_query)
        else:
            safe_limit = 200 if limit is None else max(1, min(limit, 50000))
            rows = fetch_all_dict(f"{base_query} LIMIT %s;", (safe_limit,))

        files[filename] = rows_to_csv(rows)

    return build_zip_response("source_tables_export.zip", files)


@app.route("/api/dashboard/filters")
def api_dashboard_filters():
    filters = parse_filters()
    district_where, district_params = apply_common_filters("h", {**filters, "category": "all"}, "bucket_start")
    category_where, category_params = apply_common_filters("h", {**filters, "district": "all"}, "bucket_start")

    districts = fetch_all_dict(
        f"""
        SELECT h.police_district AS value
        FROM incident_counts_hourly h
        WHERE {district_where}
          AND COALESCE(h.police_district, '') <> ''
        GROUP BY h.police_district
        ORDER BY h.police_district;
        """,
        district_params,
    )

    categories = fetch_all_dict(
        f"""
        SELECT h.incident_category AS value
        FROM incident_counts_hourly h
        WHERE {category_where}
          AND COALESCE(h.incident_category, '') <> ''
        GROUP BY h.incident_category
        ORDER BY h.incident_category;
        """,
        category_params,
    )

    return jsonify(
        {
            "status": "ok",
            "districts": [{"value": "all", "label": "All districts"}] + [
                {"value": row["value"], "label": row["value"]} for row in districts
            ],
            "categories": [{"value": "all", "label": "All categories"}] + [
                {"value": row["value"], "label": row["value"]} for row in categories
            ],
        }
    )


@app.route("/api/dashboard/overview")
def api_dashboard_overview():
    filters = parse_filters()
    is_daily = filters["window"] == "30d"
    table = "incident_counts_daily" if is_daily else "incident_counts_hourly"
    time_col = "bucket_date" if is_daily else "bucket_start"
    where_clause, params = apply_common_filters("t", filters, time_col)

    row = fetch_one_dict(
        f"""
        SELECT
            COALESCE(SUM(t.total_incidents), 0) AS total_incidents,
            CASE
                WHEN COALESCE(SUM(t.total_incidents), 0) = 0 THEN 0
                ELSE SUM(t.open_active_count)::double precision / SUM(t.total_incidents)::double precision
            END AS open_ratio,
            CASE
                WHEN COALESCE(SUM(t.total_incidents), 0) = 0 THEN 0
                ELSE SUM(t.filed_online_count)::double precision / SUM(t.total_incidents)::double precision
            END AS online_ratio
        FROM {table} t
        WHERE {where_clause};
        """,
        params,
    ) or {}

    return jsonify(
        {
            "status": "ok",
            "kpis": {
                "total_incidents": safe_int(row.get("total_incidents")),
                "open_ratio": safe_float((row.get("open_ratio") or 0) * 100),
                "online_ratio": safe_float((row.get("online_ratio") or 0) * 100),
                "avg_report_delay_minutes": 0.0,
            },
        }
    )


@app.route("/api/dashboard/trend")
def api_dashboard_trend():
    filters = parse_filters()
    is_daily = filters["window"] == "30d"
    table = "incident_counts_daily" if is_daily else "incident_counts_hourly"
    time_col = "bucket_date" if is_daily else "bucket_start"
    where_clause, params = apply_common_filters("t", filters, time_col)

    rows = fetch_all_dict(
        f"""
        SELECT
            t.{time_col} AS bucket,
            SUM(t.total_incidents) AS total_incidents,
            SUM(t.open_active_count) AS open_active_count,
            SUM(t.filed_online_count) AS filed_online_count
        FROM {table} t
        WHERE {where_clause}
        GROUP BY t.{time_col}
        ORDER BY t.{time_col};
        """,
        params,
    )

    return jsonify(
        {
            "status": "ok",
            "granularity": "daily" if is_daily else "hourly",
            "series": [
                {
                    "bucket": row["bucket"].isoformat() if row.get("bucket") else None,
                    "total_incidents": safe_int(row.get("total_incidents")),
                    "open_active_count": safe_int(row.get("open_active_count")),
                    "filed_online_count": safe_int(row.get("filed_online_count")),
                }
                for row in rows
            ],
        }
    )


@app.route("/api/dashboard/district-pressure")
def api_dashboard_district_pressure():
    filters = parse_filters()
    is_daily = filters["window"] == "30d"
    table = "incident_counts_daily" if is_daily else "incident_counts_hourly"
    time_col = "bucket_date" if is_daily else "bucket_start"
    where_clause, params = apply_common_filters("d", {**filters, "district": "all"}, time_col)

    rows = fetch_all_dict(
        f"""
        SELECT
            d.police_district,
            SUM(d.total_incidents) AS total_incidents,
            SUM(d.open_active_count) AS open_active_count,
            SUM(d.filed_online_count) AS filed_online_count
        FROM {table} d
        WHERE {where_clause}
          AND COALESCE(d.police_district, '') <> ''
        GROUP BY d.police_district
        ORDER BY total_incidents DESC, d.police_district ASC
        LIMIT 10;
        """,
        params,
    )

    districts = []
    for row in rows:
        total = max(safe_int(row.get("total_incidents")), 1)
        districts.append(
            {
                "police_district": row.get("police_district") or "Unknown",
                "total_incidents": safe_int(row.get("total_incidents")),
                "open_ratio": safe_float((safe_int(row.get("open_active_count")) / total) * 100),
                "online_ratio": safe_float((safe_int(row.get("filed_online_count")) / total) * 100),
            }
        )

    return jsonify({"status": "ok", "districts": districts})


@app.route("/api/dashboard/category-mix")
def api_dashboard_category_mix():
    filters = parse_filters()
    is_daily = filters["window"] == "30d"
    table = "incident_counts_daily" if is_daily else "incident_counts_hourly"
    time_col = "bucket_date" if is_daily else "bucket_start"
    where_clause, params = apply_common_filters("c", {**filters, "category": "all"}, time_col)

    rows = fetch_all_dict(
        f"""
        SELECT
            c.incident_category,
            SUM(c.total_incidents) AS total_incidents
        FROM {table} c
        WHERE {where_clause}
          AND COALESCE(c.incident_category, '') <> ''
        GROUP BY c.incident_category
        ORDER BY total_incidents DESC, c.incident_category ASC
        LIMIT 8;
        """,
        params,
    )

    return jsonify(
        {
            "status": "ok",
            "labels": [row["incident_category"] for row in rows],
            "values": [safe_int(row["total_incidents"]) for row in rows],
        }
    )


@app.route("/api/dashboard/risk-signals")
def api_dashboard_risk_signals():
    filters = parse_filters()
    where_clause, params = apply_common_filters("r", filters, "feature_timestamp")
    row = fetch_one_dict(
        f"""
        SELECT
            AVG(r.incidents_last_1h) AS incidents_last_1h,
            AVG(r.incidents_last_3h) AS incidents_last_3h,
            AVG(r.incidents_last_6h) AS incidents_last_6h,
            AVG(r.incidents_last_24h) AS incidents_last_24h,
            AVG(r.incidents_last_7d) AS incidents_last_7d,
            AVG(r.open_active_ratio_24h) AS open_active_ratio_24h,
            AVG(r.filed_online_ratio_24h) AS filed_online_ratio_24h,
            AVG(r.avg_report_delay_minutes_24h) AS avg_report_delay_minutes_24h
        FROM risk_features_hourly r
        WHERE {where_clause};
        """,
        params,
    ) or {}

    open_ratio_pct = safe_float((row.get("open_active_ratio_24h") or 0) * 100)
    delay_minutes = safe_float(row.get("avg_report_delay_minutes_24h"))

    signals = [
        {
            "label": "Avg incidents in last 1h",
            "value": safe_float(row.get("incidents_last_1h")),
            "severity": "high" if safe_float(row.get("incidents_last_1h")) >= 8 else "medium" if safe_float(row.get("incidents_last_1h")) >= 4 else "low",
            "description": "Rolling short-horizon pressure from risk_features_hourly.",
        },
        {
            "label": "Avg incidents in last 24h",
            "value": safe_float(row.get("incidents_last_24h")),
            "severity": "high" if safe_float(row.get("incidents_last_24h")) >= 80 else "medium" if safe_float(row.get("incidents_last_24h")) >= 30 else "low",
            "description": "Daily pressure proxy before model scoring exists.",
        },
        {
            "label": "Open / Active ratio 24h",
            "value": open_ratio_pct,
            "suffix": "%",
            "severity": "high" if open_ratio_pct >= 40 else "medium" if open_ratio_pct >= 20 else "low",
            "description": "Resolution pressure across the selected scope.",
        },
        {
            "label": "Avg report delay 24h",
            "value": delay_minutes,
            "suffix": " min",
            "severity": "high" if delay_minutes >= 120 else "medium" if delay_minutes >= 45 else "low",
            "description": "Lag between incident and report registration.",
        },
    ]

    mode = request.args.get("risk_mode", "volume")
    if mode == "open":
        signals.sort(key=lambda item: (0 if item["label"].startswith("Open") else 1, item["label"]))
    elif mode == "delay":
        signals.sort(key=lambda item: (0 if item["label"].startswith("Avg report delay") else 1, item["label"]))

    return jsonify({"status": "ok", "signals": signals})


@app.route("/api/dashboard/map-points")
def api_dashboard_map_points():
    filters = parse_filters()
    risk_mode = (request.args.get("risk_mode") or "volume").strip().lower()
    if risk_mode not in {"volume", "open", "delay"}:
        risk_mode = "volume"

    where_clause, params = apply_common_filters("r", filters, "incident_datetime")

    category_placeholders = ", ".join(["%s"] * len(CATEGORY_FILTER_VALUES))
    params = params + tuple(CATEGORY_FILTER_VALUES) + (MAP_POINT_LIMIT,)

    rows = fetch_all_dict(
        f"""
        SELECT
            r.row_id,
            r.incident_datetime,
            r.police_district,
            r.incident_category,
            r.incident_subcategory,
            r.incident_description,
            r.resolution,
            r.report_delay_minutes,
            r.latitude,
            r.longitude
        FROM incidents_raw r
        WHERE {where_clause}
          AND r.incident_category IN ({category_placeholders})
          AND r.latitude IS NOT NULL
          AND r.longitude IS NOT NULL
        ORDER BY r.incident_datetime DESC
        LIMIT %s;
        """,
        params,
    )

    points = []
    for row in rows:
        risk_score, risk_mode_label = compute_point_risk(row, risk_mode)
        points.append(
            {
                "id": row.get("row_id"),
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "police_district": row.get("police_district") or "Unknown",
                "incident_category": row.get("incident_category") or "Unknown",
                "incident_subcategory": row.get("incident_subcategory") or "Unknown",
                "incident_description": row.get("incident_description") or "No description",
                "resolution": row.get("resolution") or "Unknown",
                "incident_datetime": row["incident_datetime"].isoformat() if row.get("incident_datetime") else None,
                "risk_score": risk_score,
                "risk_level": "high" if risk_score >= 0.7 else "medium" if risk_score >= 0.4 else "low",
                "risk_mode": risk_mode,
                "risk_mode_label": risk_mode_label,
                "marker_radius": 8 if risk_score >= 0.85 else 7 if risk_score >= 0.65 else 6 if risk_score >= 0.45 else 4.5,
                "heat_weight": risk_score,
            }
        )

    return jsonify(
        {
            "status": "ok",
            "center": SF_CENTER,
            "point_count": len(points),
            "risk_mode": risk_mode,
            "points": points,
        }
    )


@app.route("/api/dashboard/forecast-training-summary")
def api_dashboard_forecast_training_summary():
    filters = parse_filters()
    where_clause, params = apply_common_filters("f", filters, "bucket_start")
    row = fetch_one_dict(
        f"""
        SELECT
            COUNT(*) AS rows_count,
            COUNT(DISTINCT f.series_id) AS series_count,
            MIN(f.bucket_start) AS min_bucket,
            MAX(f.bucket_start) AS max_bucket,
            SUM(f.total_incidents) AS total_incidents
        FROM forecast_training_series f
        WHERE {where_clause};
        """,
        params,
    ) or {}

    return jsonify(
        {
            "status": "ok",
            "rows_count": safe_int(row.get("rows_count")),
            "series_count": safe_int(row.get("series_count")),
            "total_incidents": safe_int(row.get("total_incidents")),
            "min_bucket": row.get("min_bucket").isoformat() if row.get("min_bucket") else None,
            "max_bucket": row.get("max_bucket").isoformat() if row.get("max_bucket") else None,
        }
    )








SF_TZ = ZoneInfo("America/Los_Angeles")

def parse_sf_target_timestamp(raw_value: str) -> datetime | None:
    raw_value = (raw_value or "").strip()

    if not raw_value or raw_value.lower() == "latest":
        return None

    parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))

    if parsed.tzinfo is not None:
        return parsed.astimezone(SF_TZ).replace(tzinfo=None)

    return parsed


def apply_projection_timestamp_to_rows(
    rows: list[dict[str, Any]],
    projection_timestamp: datetime | None,
) -> list[dict[str, Any]]:
    """
    Socrata/DataSF timestamps are treated as America/Los_Angeles local time.
    The volume model receives temporal features explicitly, so custom projection
    times must update hour/day/month before inference.

    For a requested projection time, feature_timestamp is set to one hour before
    so predict_volume_from_rows reports forecast_for as the requested time.
    Rolling-window features still come from the latest available feature context.
    """
    if projection_timestamp is None:
        return rows

    feature_timestamp = projection_timestamp - timedelta(hours=1)
    projected_rows: list[dict[str, Any]] = []

    for row in rows:
        projected_row = dict(row)
        projected_row["feature_timestamp"] = feature_timestamp
        projected_row["hour_of_day"] = projection_timestamp.hour
        projected_row["day_of_week"] = projection_timestamp.strftime("%A")
        projected_row["month_of_year"] = projection_timestamp.month
        projected_rows.append(projected_row)

    return projected_rows



@app.route("/admin/ml/volume/model-info", methods=["GET"])
@require_admin_token
def admin_ml_volume_model_info():
    metrics = read_volume_model_metrics()
    if not metrics:
        return jsonify({
            "status": "not_found",
            "message": "No trained volume model metrics found yet.",
            "model_name": VOLUME_MODEL_NAME,
            "model_dir": str(MODEL_DIR),
            "bucket": MODEL_BUCKET_NAME,
            "model_key": get_model_artifact_keys()["model"],
        }), 404

    return jsonify(metrics)


@app.route("/admin/ml/volume/predict", methods=["POST"])
@require_admin_token
def admin_ml_volume_predict():
    payload = parse_admin_json_payload()
    limit = parse_positive_int(payload.get("limit"), default=200, min_value=1, max_value=5000)
    district = (payload.get("district") or "").strip() or None
    category = (payload.get("category") or "").strip() or None

    raw_target_timestamp = (payload.get("target_timestamp") or "latest").strip()
    try:
        target_timestamp = parse_sf_target_timestamp(raw_target_timestamp)
    except ValueError:
        return jsonify({
            "status": "error",
            "message": (
                "Invalid target_timestamp. Use 'latest', a San Francisco local timestamp "
                "like 2026-04-24T22:00:00, or an ISO timestamp with timezone like "
                "2026-04-25T05:00:00Z."
            ),
        }), 400

    try:
        rows = fetch_latest_volume_features(
            limit=limit,
            target_timestamp=target_timestamp,
            district=district,
            category=category,
        )
        result = predict_volume_from_rows(rows)
        metrics = read_volume_model_metrics()
        model_path = get_volume_model_path()

        return jsonify(
            {
                "status": "ok",
                "model_name": VOLUME_MODEL_NAME,
                "model_path": str(model_path),
                "model_source": result.get("model_source"),
                "model_generated_at": metrics.get("generated_at") if metrics else None,
                "filters": {
                    "target_timestamp": target_timestamp.isoformat() if target_timestamp else "latest",
                    "district": district or "all",
                    "category": category or "all",
                    "limit": limit,
                },
                **result,
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/admin/etl/run", methods=["POST"])
@require_admin_token
def admin_etl_run():
    payload = parse_admin_json_payload()
    command_name = (payload.get("command") or "").strip()
    extra_env = payload.get("env") or {}
    timeout_seconds = parse_positive_int(
        payload.get("timeout_seconds"),
        default=min(ADMIN_ETL_MAX_TIMEOUT_SECONDS, 1800),
        min_value=1,
        max_value=ADMIN_ETL_MAX_TIMEOUT_SECONDS,
    )

    if not isinstance(extra_env, dict):
        return jsonify({"status": "error", "message": "env must be a JSON object."}), 400

    try:
        result = run_allowed_admin_etl(
            command_name=command_name,
            extra_env=extra_env,
            timeout_seconds=timeout_seconds,
        )
        return jsonify({"status": "ok", **result})
    except RuntimeError as exc:
        message = str(exc)
        try:
            details = json.loads(message)
            return jsonify({"status": "error", "message": "ETL command failed.", "details": details}), 500
        except Exception:
            return jsonify({"status": "error", "message": message}), 500
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route("/api/dashboard/ml-volume-polygons", methods=["GET"])
def api_dashboard_ml_volume_polygons():
    category = (request.args.get("category") or "all").strip()
    district = (request.args.get("district") or "all").strip()

    category_filter = None if category.lower() == "all" else category
    district_filter = None if district.lower() == "all" else district

    raw_target_timestamp = (request.args.get("target_timestamp") or "latest").strip()

    try:
        target_timestamp = parse_sf_target_timestamp(raw_target_timestamp)
    except ValueError:
        return jsonify(
            {
                "status": "error",
                "message": (
                    "Invalid target_timestamp. Use 'latest', a San Francisco local timestamp "
                    "like 2026-04-26T13:00:00, or an ISO timestamp with timezone like "
                    "2026-04-26T20:00:00Z."
                ),
            }
        ), 400

    try:
        # If target_timestamp is provided, use the latest rolling feature context
        # and override only the temporal features to simulate the selected SF time.
        # If target_timestamp is latest/None, preserve the original latest-feature behavior.
        rows = fetch_latest_volume_features(
            limit=None,
            target_timestamp=None if target_timestamp else None,
            district=district_filter,
            category=category_filter,
        )

        rows = apply_projection_timestamp_to_rows(rows, target_timestamp)

        prediction_result = predict_volume_from_rows(rows)
        forecast_geojson = build_volume_forecast_geojson(
            prediction_result.get("predictions", [])
        )

        return jsonify(
            {
                "status": "ok",
                "model_name": VOLUME_MODEL_NAME,
                "model_runtime_type": prediction_result.get("model_runtime_type"),
                "model_source": prediction_result.get("model_source"),
                "filters": {
                    "target_timestamp": (
                        target_timestamp.isoformat()
                        if target_timestamp
                        else "latest"
                    ),
                    "target_timezone": "America/Los_Angeles",
                    "district": district_filter or "all",
                    "category": category_filter or "all",
                },
                "summary": {
                    "source_rows": len(rows),
                    "prediction_rows": prediction_result.get("row_count", 0),
                    "mapped_districts": forecast_geojson["mapped_districts"],
                    "total_predicted_incidents_next_hour": forecast_geojson[
                        "total_predicted_incidents_next_hour"
                    ],
                },
                "type": forecast_geojson["type"],
                "features": forecast_geojson["features"],
            }
        )

    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500



@app.route("/admin/ml/risk/model-info", methods=["GET"])
@require_admin_token
def admin_ml_risk_model_info():
    metrics = read_risk_model_metrics()
    if not metrics:
        return jsonify({
            "status": "not_found",
            "message": "No trained risk classifier model metrics found yet.",
            "model_name": RISK_CLASSIFIER_MODEL_NAME,
            "model_dir": str(MODEL_DIR),
            "bucket": MODEL_BUCKET_NAME,
            "model_key": get_risk_model_artifact_keys()["model"],
        }), 404

    return jsonify(metrics)


@app.route("/admin/ml/risk/predict", methods=["POST"])
@require_admin_token
def admin_ml_risk_predict():
    payload = parse_admin_json_payload()
    limit = parse_positive_int(payload.get("limit"), default=200, min_value=1, max_value=5000)
    district = (payload.get("district") or "").strip() or None
    category = (payload.get("category") or "").strip() or None

    raw_target_timestamp = (payload.get("target_timestamp") or "latest").strip()
    try:
        target_timestamp = parse_sf_target_timestamp(raw_target_timestamp)
    except ValueError:
        return jsonify({
            "status": "error",
            "message": (
                "Invalid target_timestamp. Use 'latest', a San Francisco local timestamp "
                "like 2026-04-24T22:00:00, or an ISO timestamp with timezone like "
                "2026-04-25T05:00:00Z."
            ),
        }), 400

    try:
        rows = fetch_latest_volume_features(
            limit=limit,
            target_timestamp=None,
            district=district,
            category=category,
        )
        rows = apply_projection_timestamp_to_rows(rows, target_timestamp)

        result = predict_risk_from_rows(rows)
        metrics = read_risk_model_metrics()
        model_path = get_risk_model_path()

        return jsonify(
            {
                "status": "ok",
                "model_name": RISK_CLASSIFIER_MODEL_NAME,
                "model_path": str(model_path),
                "model_source": result.get("model_source"),
                "model_generated_at": metrics.get("generated_at") if metrics else None,
                "filters": {
                    "target_timestamp": target_timestamp.isoformat() if target_timestamp else "latest",
                    "target_timezone": "America/Los_Angeles",
                    "district": district or "all",
                    "category": category or "all",
                    "limit": limit,
                },
                **result,
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/dashboard/ml-risk-predictions", methods=["GET"])
def api_dashboard_ml_risk_predictions():
    category = (request.args.get("category") or "all").strip()
    district = (request.args.get("district") or "all").strip()

    category_filter = None if category.lower() == "all" else category
    district_filter = None if district.lower() == "all" else district

    raw_target_timestamp = (request.args.get("target_timestamp") or "latest").strip()

    try:
        target_timestamp = parse_sf_target_timestamp(raw_target_timestamp)
    except ValueError:
        return jsonify(
            {
                "status": "error",
                "message": (
                    "Invalid target_timestamp. Use 'latest', a San Francisco local timestamp "
                    "like 2026-04-26T13:00:00, or an ISO timestamp with timezone like "
                    "2026-04-26T20:00:00Z."
                ),
            }
        ), 400

    try:
        rows = fetch_latest_volume_features(
            limit=None,
            target_timestamp=None,
            district=district_filter,
            category=category_filter,
        )

        rows = apply_projection_timestamp_to_rows(rows, target_timestamp)

        prediction_result = predict_risk_from_rows(rows)

        return jsonify(
            {
                "status": "ok",
                "model_name": RISK_CLASSIFIER_MODEL_NAME,
                "model_runtime_type": prediction_result.get("model_runtime_type"),
                "model_source": prediction_result.get("model_source"),
                "filters": {
                    "target_timestamp": target_timestamp.isoformat() if target_timestamp else "latest",
                    "target_timezone": "America/Los_Angeles",
                    "district": district_filter or "all",
                    "category": category_filter or "all",
                },
                "summary": {
                    "source_rows": len(rows),
                    "prediction_rows": prediction_result.get("row_count", 0),
                },
                "predictions": prediction_result.get("predictions", []),
            }
        )

    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/dashboard/ml-risk-polygons", methods=["GET"])
def api_dashboard_ml_risk_polygons():
    category = (request.args.get("category") or "all").strip()
    district = (request.args.get("district") or "all").strip()

    category_filter = None if category.lower() == "all" else category
    district_filter = None if district.lower() == "all" else district

    raw_target_timestamp = (request.args.get("target_timestamp") or "latest").strip()

    try:
        target_timestamp = parse_sf_target_timestamp(raw_target_timestamp)
    except ValueError:
        return jsonify(
            {
                "status": "error",
                "message": (
                    "Invalid target_timestamp. Use 'latest', a San Francisco local timestamp "
                    "like 2026-04-26T13:00:00, or an ISO timestamp with timezone like "
                    "2026-04-26T20:00:00Z."
                ),
            }
        ), 400

    try:
        rows = fetch_latest_volume_features(
            limit=None,
            target_timestamp=None,
            district=district_filter,
            category=category_filter,
        )

        rows = apply_projection_timestamp_to_rows(rows, target_timestamp)

        prediction_result = predict_risk_from_rows(rows)
        forecast_geojson = build_risk_forecast_geojson(
            prediction_result.get("predictions", [])
        )

        return jsonify(
            {
                "status": "ok",
                "model_name": RISK_CLASSIFIER_MODEL_NAME,
                "model_runtime_type": prediction_result.get("model_runtime_type"),
                "model_source": prediction_result.get("model_source"),
                "filters": {
                    "target_timestamp": target_timestamp.isoformat() if target_timestamp else "latest",
                    "target_timezone": "America/Los_Angeles",
                    "district": district_filter or "all",
                    "category": category_filter or "all",
                },
                "summary": {
                    "source_rows": len(rows),
                    "prediction_rows": prediction_result.get("row_count", 0),
                    "mapped_districts": forecast_geojson["mapped_districts"],
                    "max_risk_score": forecast_geojson["max_risk_score"],
                    "avg_risk_score": forecast_geojson["avg_risk_score"],
                },
                "type": forecast_geojson["type"],
                "features": forecast_geojson["features"],
            }
        )

    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
