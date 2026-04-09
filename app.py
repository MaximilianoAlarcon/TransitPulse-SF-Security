import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from psycopg2 import sql
import psycopg2

from utils import get_db_connection, execute_sql_file, execute_query

BASE_DIR = Path(__file__).resolve().parent

DB_CONFIG = {
    "host": os.environ.get("DB_HOST"),
    "database": os.environ.get("DB_NAME"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASSWORD"),
    "port": os.environ.get("DB_PORT")
}

app = Flask(__name__, static_folder="static")


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
    output_format = (payload.get("format") or "json").lower()
    title = payload.get("title") or "Query Result"

    try:
        result = execute_query(DB_CONFIG, query)

        if result["has_result_set"]:
            if output_format == "html":
                return render_html_table(result["columns"], result["rows"], title=title)

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


@app.route("/drop-legacy-otp-tables", methods=["POST"])
def drop_legacy_otp_tables():
    query = """
CREATE INDEX IF NOT EXISTS idx_incidents_raw_incident_datetime
ON incidents_raw (incident_datetime);

CREATE INDEX IF NOT EXISTS idx_incidents_raw_incident_date
ON incidents_raw (incident_date);

CREATE INDEX IF NOT EXISTS idx_incidents_raw_district_category_datetime
ON incidents_raw (police_district, incident_category, incident_datetime);

CREATE INDEX IF NOT EXISTS idx_incident_counts_hourly_key
ON incident_counts_hourly (bucket_start, police_district, incident_category);

CREATE INDEX IF NOT EXISTS idx_incidents_raw_delay_key
ON incidents_raw (incident_datetime, police_district, incident_category);

CREATE INDEX IF NOT EXISTS idx_forecast_training_series_bucket
ON forecast_training_series (bucket_start);

CREATE INDEX IF NOT EXISTS idx_risk_features_hourly_ts
ON risk_features_hourly (feature_timestamp);
    """

    try:
        result = execute_query(DB_CONFIG, query)
        return jsonify(
            {
                "status": "ok",
                "message": "Legacy OTP tables were dropped if they existed.",
                "query": query,
                "result_type": "command",
                "affected_rows": result.get("affected_rows"),
                "status_message": result.get("status_message"),
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc), "query": query}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
