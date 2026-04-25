import csv
import io
import json
import math
import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
from typing import Any, Callable

from flask import Flask, Response, jsonify, render_template, request
from psycopg2.extras import RealDictCursor

from utils import execute_query, execute_sql_file, get_db_connection

from zoneinfo import ZoneInfo


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

MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "models"))
VOLUME_MODEL_NAME = os.environ.get("VOLUME_MODEL_NAME", "volume_random_forest_v1")
DEFAULT_ML_LOOKBACK_DAYS = int(os.environ.get("ML_LOOKBACK_DAYS", "180"))
DEFAULT_ML_TEST_SIZE = float(os.environ.get("ML_TEST_SIZE", "0.2"))
DEFAULT_ML_MIN_ROWS = int(os.environ.get("ML_MIN_ROWS", "200"))

ML_NUMERIC_FEATURES = [
    "hour_of_day",
    "month_of_year",
    "incidents_last_1h",
    "incidents_last_3h",
    "incidents_last_6h",
    "incidents_last_24h",
    "incidents_last_7d",
    "open_active_ratio_24h",
    "filed_online_ratio_24h",
    "avg_report_delay_minutes_24h",
]

ML_CATEGORICAL_FEATURES = [
    "police_district",
    "incident_category",
    "day_of_week",
]

ML_TARGET_COLUMN = "target_incidents_next_hour"


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


def fetch_volume_ml_dataset(lookback_days: int) -> list[dict[str, Any]]:
    return fetch_all_dict(
        """
        WITH next_counts AS (
            SELECT
                bucket_start,
                police_district,
                incident_category,
                SUM(total_incidents) AS total_incidents
            FROM incident_counts_hourly
            GROUP BY 1,2,3
        ), bounds AS (
            SELECT MAX(bucket_start) AS max_bucket_start
            FROM incident_counts_hourly
        )
        SELECT
            f.feature_timestamp,
            f.police_district,
            f.incident_category,
            f.hour_of_day,
            f.day_of_week,
            f.month_of_year,
            COALESCE(f.incidents_last_1h, 0) AS incidents_last_1h,
            COALESCE(f.incidents_last_3h, 0) AS incidents_last_3h,
            COALESCE(f.incidents_last_6h, 0) AS incidents_last_6h,
            COALESCE(f.incidents_last_24h, 0) AS incidents_last_24h,
            COALESCE(f.incidents_last_7d, 0) AS incidents_last_7d,
            COALESCE(f.open_active_ratio_24h, 0) AS open_active_ratio_24h,
            COALESCE(f.filed_online_ratio_24h, 0) AS filed_online_ratio_24h,
            COALESCE(f.avg_report_delay_minutes_24h, 0) AS avg_report_delay_minutes_24h,
            COALESCE(n.total_incidents, 0) AS target_incidents_next_hour
        FROM risk_features_hourly f
        CROSS JOIN bounds b
        LEFT JOIN next_counts n
            ON n.bucket_start = f.feature_timestamp + INTERVAL '1 hour'
           AND n.police_district = f.police_district
           AND n.incident_category = f.incident_category
        WHERE f.feature_timestamp >= NOW() - (%s::int * INTERVAL '1 day')
          AND f.feature_timestamp < b.max_bucket_start
          AND COALESCE(f.police_district, '') <> ''
          AND COALESCE(f.incident_category, '') <> ''
        ORDER BY f.feature_timestamp ASC, f.police_district ASC, f.incident_category ASC;
        """,
        (lookback_days,),
    )


def summarize_volume_ml_dataset(rows: list[dict[str, Any]], lookback_days: int) -> dict[str, Any]:
    if not rows:
        return {"lookback_days": lookback_days, "row_count": 0, "min_feature_timestamp": None, "max_feature_timestamp": None, "district_count": 0, "category_count": 0, "target_sum": 0, "target_avg": 0, "target_max": 0}

    timestamps = [row["feature_timestamp"] for row in rows if row.get("feature_timestamp")]
    targets = [float(row.get(ML_TARGET_COLUMN) or 0) for row in rows]
    districts = {row.get("police_district") for row in rows if row.get("police_district")}
    categories = {row.get("incident_category") for row in rows if row.get("incident_category")}

    return {
        "lookback_days": lookback_days,
        "row_count": len(rows),
        "min_feature_timestamp": min(timestamps).isoformat() if timestamps else None,
        "max_feature_timestamp": max(timestamps).isoformat() if timestamps else None,
        "district_count": len(districts),
        "category_count": len(categories),
        "target_sum": round(sum(targets), 4),
        "target_avg": round(sum(targets) / len(targets), 4) if targets else 0,
        "target_max": round(max(targets), 4) if targets else 0,
    }


def train_volume_forecast_model(rows: list[dict[str, Any]], test_size: float) -> dict[str, Any]:
    try:
        import joblib
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except ImportError as exc:
        raise RuntimeError("Missing ML dependencies. Add scikit-learn, pandas, and joblib to requirements.txt.") from exc

    if len(rows) < DEFAULT_ML_MIN_ROWS:
        raise ValueError(f"Not enough rows to train. Required at least {DEFAULT_ML_MIN_ROWS}, got {len(rows)}.")

    df = pd.DataFrame(rows).sort_values("feature_timestamp").reset_index(drop=True)

    for column in ML_NUMERIC_FEATURES + [ML_TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    for column in ML_CATEGORICAL_FEATURES:
        df[column] = df[column].fillna("Unknown").astype(str)

    feature_columns = ML_NUMERIC_FEATURES + ML_CATEGORICAL_FEATURES
    X = df[feature_columns]
    y = df[ML_TARGET_COLUMN]

    split_index = int(len(df) * (1.0 - test_size))
    split_index = max(1, min(split_index, len(df) - 1))

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", encoder, ML_CATEGORICAL_FEATURES),
            ("numeric", "passthrough", ML_NUMERIC_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=150, max_depth=14, min_samples_leaf=2, random_state=42, n_jobs=-1)),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, predictions))
    mse = float(mean_squared_error(y_test, predictions))
    rmse = math.sqrt(mse)
    r2 = float(r2_score(y_test, predictions)) if len(y_test) > 1 else 0.0

    actual_sum = float(y_test.sum())
    predicted_sum = float(predictions.sum())
    aggregate_error_pct = abs(predicted_sum - actual_sum) / max(actual_sum, 1.0) * 100.0

    generated_at = datetime.utcnow().replace(microsecond=0)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"{VOLUME_MODEL_NAME}.joblib"
    metrics_path = MODEL_DIR / f"{VOLUME_MODEL_NAME}_metrics.json"

    joblib.dump(model, model_path)

    metrics = {
        "status": "ok",
        "model_name": VOLUME_MODEL_NAME,
        "model_type": "RandomForestRegressor",
        "generated_at": generated_at.isoformat(),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "features": feature_columns,
        "target": ML_TARGET_COLUMN,
        "row_count": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": test_size,
        "time_range": {
            "min_feature_timestamp": df["feature_timestamp"].min().isoformat(),
            "max_feature_timestamp": df["feature_timestamp"].max().isoformat(),
        },
        "metrics": {
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "r2": round(r2, 6),
            "actual_sum_test": round(actual_sum, 6),
            "predicted_sum_test": round(predicted_sum, 6),
            "aggregate_error_pct": round(aggregate_error_pct, 6),
        },
    }

    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return metrics


def read_volume_model_metrics() -> dict[str, Any] | None:
    metrics_path = MODEL_DIR / f"{VOLUME_MODEL_NAME}_metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


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


@app.route("/admin/ml/volume/dataset-summary", methods=["POST"])
@require_admin_token
def admin_ml_volume_dataset_summary():
    payload = parse_admin_json_payload()
    lookback_days = parse_positive_int(payload.get("lookback_days"), default=DEFAULT_ML_LOOKBACK_DAYS, min_value=7, max_value=3650)

    rows = fetch_volume_ml_dataset(lookback_days=lookback_days)
    summary = summarize_volume_ml_dataset(rows, lookback_days=lookback_days)

    return jsonify({"status": "ok", "dataset": summary})


@app.route("/admin/ml/volume/train", methods=["POST"])
@require_admin_token
def admin_ml_volume_train():
    payload = parse_admin_json_payload()
    lookback_days = parse_positive_int(payload.get("lookback_days"), default=DEFAULT_ML_LOOKBACK_DAYS, min_value=7, max_value=3650)
    test_size = parse_float_range(payload.get("test_size"), default=DEFAULT_ML_TEST_SIZE, min_value=0.05, max_value=0.4)

    try:
        rows = fetch_volume_ml_dataset(lookback_days=lookback_days)
        dataset_summary = summarize_volume_ml_dataset(rows, lookback_days=lookback_days)
        training_result = train_volume_forecast_model(rows, test_size=test_size)
        return jsonify({"status": "ok", "dataset": dataset_summary, "training": training_result})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/admin/ml/volume/model-info", methods=["GET"])
@require_admin_token
def admin_ml_volume_model_info():
    metrics = read_volume_model_metrics()
    if not metrics:
        return jsonify({"status": "not_found", "message": "No trained volume model metrics found yet.", "model_name": VOLUME_MODEL_NAME, "model_dir": str(MODEL_DIR)}), 404

    return jsonify(metrics)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
