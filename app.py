import os
from pathlib import Path

from flask import Flask, jsonify, render_template
from psycopg2 import sql
import psycopg2

from utils import get_db_connection, execute_sql_file

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
        with get_db_connection(DB_CONFIG) as conn:
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
