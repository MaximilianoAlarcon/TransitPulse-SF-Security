import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request
import psycopg2
from psycopg2.extras import RealDictCursor

from utils import get_db_connection, execute_query, execute_sql_file

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


def fetch_all_dict(query: str, params: tuple[Any, ...] = ()):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]


def fetch_one_dict(query: str, params: tuple[Any, ...] = ()):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            row = cur.fetchone()
            return dict(row) if row else None


def safe_int(value: Any) -> int:
    if value is None:
        return 0
    return int(value)


def safe_float(value: Any, digits: int = 2) -> float:
    if value is None:
        return 0.0
    return round(float(value), digits)


def parse_filters():
    window = request.args.get("window", "7d")
    district = (request.args.get("district") or "all").strip()
    category = (request.args.get("category") or "all").strip()

    if window not in WINDOW_TO_DELTA:
        window = "7d"

    end_dt = datetime.utcnow()
    start_dt = end_dt - WINDOW_TO_DELTA[window]

    district = "all" if not district else district
    category = "all" if not category else category

    return {
        "window": window,
        "district": district,
        "category": category,
        "start_dt": start_dt,
        "end_dt": end_dt,
    }


def apply_common_filters(alias: str, filters: dict[str, Any], timestamp_column: str):
    conditions = [f"{alias}.{timestamp_column} >= %s", f"{alias}.{timestamp_column} < %s"]
    params: list[Any] = [filters["start_dt"], filters["end_dt"]]

    if filters["district"].lower() != "all":
        conditions.append(f"{alias}.police_district = %s")
        params.append(filters["district"])

    if filters["category"].lower() != "all":
        conditions.append(f"{alias}.incident_category = %s")
        params.append(filters["category"])

    return " AND ".join(conditions), tuple(params)


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
                "sql_file": str(sql_file.name),
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/db-tables")
def db_tables():
    query = """
        SELECT
            t.table_schema,
            t.table_name,
            COALESCE(s.n_live_tup::bigint, 0) AS estimated_rows,
            obj_description((quote_ident(t.table_schema) || '.' || quote_ident(t.table_name))::regclass, 'pg_class') AS table_comment
        FROM information_schema.tables AS t
        LEFT JOIN pg_stat_user_tables AS s
            ON s.schemaname = t.table_schema
           AND s.relname = t.table_name
        WHERE t.table_type = 'BASE TABLE'
          AND t.table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY t.table_schema, t.table_name;
    """

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()

        tables = [
            {
                "schema": row[0],
                "table_name": row[1],
                "estimated_rows": row[2],
                "comment": row[3],
            }
            for row in rows
        ]

        return jsonify(
            {
                "status": "ok",
                "database": DB_CONFIG.get("database"),
                "total_tables": len(tables),
                "tables": tables,
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


@app.route("/api/dashboard/filters")
def api_dashboard_filters():
    filters = parse_filters()

    district_where, district_params = apply_common_filters("h", {**filters, "category": "all"}, "bucket_start")
    category_where, category_params = apply_common_filters("h", {**filters, "district": "all"}, "bucket_start")

    districts = fetch_all_dict(
        f"""
        SELECT h.police_district AS value, COUNT(*) AS rows_count
        FROM incident_counts_hourly h
        WHERE {district_where}
          AND h.police_district IS NOT NULL
          AND h.police_district <> ''
        GROUP BY h.police_district
        ORDER BY h.police_district;
        """,
        district_params,
    )

    categories = fetch_all_dict(
        f"""
        SELECT h.incident_category AS value, COUNT(*) AS rows_count
        FROM incident_counts_hourly h
        WHERE {category_where}
          AND h.incident_category IS NOT NULL
          AND h.incident_category <> ''
        GROUP BY h.incident_category
        ORDER BY h.incident_category;
        """,
        category_params,
    )

    return jsonify(
        {
            "status": "ok",
            "filters": {
                "window": filters["window"],
                "district": filters["district"],
                "category": filters["category"],
            },
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
    where_clause, params = apply_common_filters("r", filters, "incident_datetime")

    row = fetch_one_dict(
        f"""
        SELECT
            COUNT(*) AS total_incidents,
            AVG(CASE WHEN COALESCE(r.resolution, '') ILIKE 'Open%%' OR COALESCE(r.resolution, '') ILIKE 'Active%%' THEN 1.0 ELSE 0.0 END) AS open_ratio,
            AVG(CASE WHEN r.filed_online THEN 1.0 ELSE 0.0 END) AS online_ratio,
            AVG(r.report_delay_minutes) AS avg_report_delay_minutes
        FROM incidents_raw r
        WHERE {where_clause};
        """,
        params,
    ) or {}

    return jsonify(
        {
            "status": "ok",
            "filters": {
                "window": filters["window"],
                "district": filters["district"],
                "category": filters["category"],
            },
            "kpis": {
                "total_incidents": safe_int(row.get("total_incidents")),
                "open_ratio": safe_float((row.get("open_ratio") or 0) * 100),
                "online_ratio": safe_float((row.get("online_ratio") or 0) * 100),
                "avg_report_delay_minutes": safe_float(row.get("avg_report_delay_minutes")),
            },
        }
    )


@app.route("/api/dashboard/trend")
def api_dashboard_trend():
    filters = parse_filters()
    is_daily = filters["window"] == "30d"
    table = "incident_counts_daily" if is_daily else "incident_counts_hourly"
    time_col = "bucket_date" if is_daily else "bucket_start"
    alias = "t"
    where_clause, params = apply_common_filters(alias, filters, time_col)

    rows = fetch_all_dict(
        f"""
        SELECT
            {alias}.{time_col} AS bucket,
            SUM({alias}.total_incidents) AS total_incidents,
            SUM({alias}.open_active_count) AS open_active_count,
            SUM({alias}.filed_online_count) AS filed_online_count
        FROM {table} {alias}
        WHERE {where_clause}
        GROUP BY {alias}.{time_col}
        ORDER BY {alias}.{time_col};
        """,
        params,
    )

    series = [
        {
            "bucket": row["bucket"].isoformat() if row["bucket"] else None,
            "total_incidents": safe_int(row["total_incidents"]),
            "open_active_count": safe_int(row["open_active_count"]),
            "filed_online_count": safe_int(row["filed_online_count"]),
        }
        for row in rows
    ]

    return jsonify(
        {
            "status": "ok",
            "granularity": "daily" if is_daily else "hourly",
            "series": series,
        }
    )


@app.route("/api/dashboard/district-pressure")
def api_dashboard_district_pressure():
    filters = parse_filters()
    table = "incident_counts_daily" if filters["window"] == "30d" else "incident_counts_hourly"
    time_col = "bucket_date" if filters["window"] == "30d" else "bucket_start"
    alias = "d"
    where_clause, params = apply_common_filters(alias, {**filters, "district": "all"}, time_col)

    rows = fetch_all_dict(
        f"""
        SELECT
            d.police_district,
            SUM(d.total_incidents) AS total_incidents,
            SUM(d.open_active_count) AS open_active_count,
            SUM(d.filed_online_count) AS filed_online_count
        FROM {table} d
        WHERE {where_clause}
          AND d.police_district IS NOT NULL
          AND d.police_district <> ''
        GROUP BY d.police_district
        ORDER BY total_incidents DESC, d.police_district ASC
        LIMIT 10;
        """,
        params,
    )

    result = []
    for row in rows:
        total = max(safe_int(row["total_incidents"]), 1)
        open_ratio = safe_float((safe_int(row["open_active_count"]) / total) * 100)
        result.append(
            {
                "police_district": row["police_district"],
                "total_incidents": safe_int(row["total_incidents"]),
                "open_ratio": open_ratio,
                "online_ratio": safe_float((safe_int(row["filed_online_count"]) / total) * 100),
            }
        )

    return jsonify({"status": "ok", "districts": result})


@app.route("/api/dashboard/category-mix")
def api_dashboard_category_mix():
    filters = parse_filters()
    table = "incident_counts_daily" if filters["window"] == "30d" else "incident_counts_hourly"
    time_col = "bucket_date" if filters["window"] == "30d" else "bucket_start"
    alias = "c"
    where_clause, params = apply_common_filters(alias, {**filters, "category": "all"}, time_col)

    rows = fetch_all_dict(
        f"""
        SELECT
            c.incident_category,
            SUM(c.total_incidents) AS total_incidents
        FROM {table} c
        WHERE {where_clause}
          AND c.incident_category IS NOT NULL
          AND c.incident_category <> ''
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

    mode = request.args.get("risk_mode", "volume")
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
            "description": "Daily pressure proxy, useful before model scoring exists.",
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

    if mode == "open":
        signals.sort(key=lambda x: (0 if x["label"].startswith("Open") else 1, x["label"]))
    elif mode == "delay":
        signals.sort(key=lambda x: (0 if x["label"].startswith("Avg report delay") else 1, x["label"]))

    return jsonify({"status": "ok", "signals": signals})


@app.route("/api/dashboard/map-points")
def api_dashboard_map_points():
    filters = parse_filters()
    where_clause, params = apply_common_filters("r", filters, "incident_datetime")
    params = params + (MAP_POINT_LIMIT,)

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
            r.latitude,
            r.longitude
        FROM incidents_raw r
        WHERE {where_clause}
          AND r.latitude IS NOT NULL
          AND r.longitude IS NOT NULL
        ORDER BY r.incident_datetime DESC
        LIMIT %s;
        """,
        params,
    )

    points = []
    for row in rows:
        category = row.get("incident_category") or "Unknown"
        points.append(
            {
                "id": row.get("row_id"),
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "police_district": row.get("police_district") or "Unknown",
                "incident_category": category,
                "incident_subcategory": row.get("incident_subcategory") or "Unknown",
                "incident_description": row.get("incident_description") or "No description",
                "resolution": row.get("resolution") or "Unknown",
                "incident_datetime": row["incident_datetime"].isoformat() if row.get("incident_datetime") else None,
                "risk_level": "high" if any(word in category.lower() for word in ["assault", "robbery", "burglary", "weapon"]) else "medium" if any(word in category.lower() for word in ["theft", "larceny", "vandalism"]) else "low",
            }
        )

    return jsonify(
        {
            "status": "ok",
            "center": SF_CENTER,
            "point_count": len(points),
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
