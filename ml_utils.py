import io
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from psycopg2.extras import RealDictCursor

from utils import get_db_connection


BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "models"))
VOLUME_MODEL_NAME = os.environ.get("VOLUME_MODEL_NAME", "volume_random_forest_v1")

MODEL_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME")
MODEL_BUCKET_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL")
MODEL_BUCKET_REGION = os.environ.get("AWS_DEFAULT_REGION", "auto")
MODEL_S3_PREFIX = os.environ.get("MODEL_S3_PREFIX", "models")

POLICE_DISTRICTS_GEOJSON_PATH = BASE_DIR / "static" / "sf_police_districts_polygons.geojson"

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

VALID_MAP_DISTRICTS = {
    "BAYVIEW",
    "CENTRAL",
    "INGLESIDE",
    "MISSION",
    "NORTHERN",
    "PARK",
    "RICHMOND",
    "SOUTHERN",
    "TARAVAL",
    "TENDERLOIN",
}

VOLUME_MODEL_CACHE: dict[str, Any] = {
    "model": None,
    "source": None,
}


def fetch_all_dict(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]


def get_volume_model_path() -> Path:
    return MODEL_DIR / f"{VOLUME_MODEL_NAME}.joblib"


def get_model_artifact_keys() -> dict[str, str]:
    return {
        "model": f"{MODEL_S3_PREFIX.rstrip('/')}/{VOLUME_MODEL_NAME}.joblib",
        "metrics": f"{MODEL_S3_PREFIX.rstrip('/')}/{VOLUME_MODEL_NAME}_metrics.json",
    }


def get_s3_client() -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Missing S3 dependency. Add boto3 to requirements.txt.") from exc

    if not MODEL_BUCKET_NAME:
        raise RuntimeError("AWS_S3_BUCKET_NAME is not configured.")

    if not MODEL_BUCKET_ENDPOINT_URL:
        raise RuntimeError("AWS_ENDPOINT_URL is not configured.")

    return boto3.client(
        "s3",
        endpoint_url=MODEL_BUCKET_ENDPOINT_URL,
        region_name=MODEL_BUCKET_REGION,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def upload_bytes_to_model_bucket(key: str, data: bytes, content_type: str) -> None:
    client = get_s3_client()
    client.put_object(
        Bucket=MODEL_BUCKET_NAME,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def download_bytes_from_model_bucket(key: str) -> bytes:
    client = get_s3_client()
    response = client.get_object(Bucket=MODEL_BUCKET_NAME, Key=key)
    return response["Body"].read()


def upload_volume_model_artifacts(model_path: Path, metrics_path: Path) -> dict[str, Any]:
    keys = get_model_artifact_keys()

    if not model_path.exists():
        raise FileNotFoundError(f"Local model artifact not found at {model_path}.")

    if not metrics_path.exists():
        raise FileNotFoundError(f"Local metrics artifact not found at {metrics_path}.")

    upload_bytes_to_model_bucket(
        key=keys["model"],
        data=model_path.read_bytes(),
        content_type="application/octet-stream",
    )

    upload_bytes_to_model_bucket(
        key=keys["metrics"],
        data=metrics_path.read_bytes(),
        content_type="application/json",
    )

    return {
        "bucket": MODEL_BUCKET_NAME,
        "endpoint_url": MODEL_BUCKET_ENDPOINT_URL,
        "model_key": keys["model"],
        "metrics_key": keys["metrics"],
    }


def read_volume_model_metrics_from_bucket() -> dict[str, Any] | None:
    keys = get_model_artifact_keys()

    try:
        raw = download_bytes_from_model_bucket(keys["metrics"])
    except Exception:
        return None

    return json.loads(raw.decode("utf-8"))


def read_volume_model_metrics() -> dict[str, Any] | None:
    metrics_path = MODEL_DIR / f"{VOLUME_MODEL_NAME}_metrics.json"

    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    if MODEL_BUCKET_NAME and MODEL_BUCKET_ENDPOINT_URL:
        return read_volume_model_metrics_from_bucket()

    return None


def load_volume_model_from_bucket() -> Any:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Missing ML dependency. Add joblib to requirements.txt.") from exc

    keys = get_model_artifact_keys()
    raw = download_bytes_from_model_bucket(keys["model"])
    return joblib.load(io.BytesIO(raw))


def load_volume_model() -> tuple[Any, str]:
    cached_model = VOLUME_MODEL_CACHE.get("model")
    cached_source = VOLUME_MODEL_CACHE.get("source")

    if cached_model is not None:
        return cached_model, str(cached_source or "memory_cache")

    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Missing ML dependency. Add joblib to requirements.txt.") from exc

    model_path = get_volume_model_path()

    if model_path.exists():
        model = joblib.load(model_path)
        VOLUME_MODEL_CACHE["model"] = model
        VOLUME_MODEL_CACHE["source"] = str(model_path)
        return model, str(model_path)

    model = load_volume_model_from_bucket()
    VOLUME_MODEL_CACHE["model"] = model
    VOLUME_MODEL_CACHE["source"] = f"s3://{MODEL_BUCKET_NAME}/{get_model_artifact_keys()['model']}"

    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
    except Exception:
        pass

    return model, str(VOLUME_MODEL_CACHE["source"])


def fetch_latest_volume_features(
    limit: int | None = None,
    target_timestamp: datetime | None = None,
    district: str | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    conditions = [
        "f.feature_timestamp = COALESCE(%s::timestamp, (SELECT MAX(feature_timestamp) FROM risk_features_hourly))"
    ]
    params: list[Any] = [target_timestamp]

    if district:
        conditions.append("f.police_district = %s")
        params.append(district)

    if category:
        conditions.append("f.incident_category = %s")
        params.append(category)

    limit_clause = ""
    if limit is not None:
        limit_clause = "LIMIT %s"
        params.append(limit)

    return fetch_all_dict(
        f"""
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
            COALESCE(f.avg_report_delay_minutes_24h, 0) AS avg_report_delay_minutes_24h
        FROM risk_features_hourly f
        WHERE {' AND '.join(conditions)}
          AND COALESCE(f.police_district, '') <> ''
          AND COALESCE(f.incident_category, '') <> ''
        ORDER BY f.police_district ASC, f.incident_category ASC
        {limit_clause};
        """,
        tuple(params),
    )


def predict_volume_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Missing ML dependency. Add pandas to requirements.txt.") from exc

    if not rows:
        return {
            "predictions": [],
            "row_count": 0,
            "total_predicted_incidents_next_hour": 0.0,
            "model_source": None,
            "model_runtime_type": None,
        }

    try:
        model, model_source = load_volume_model()
    except Exception as exc:
        raise FileNotFoundError(
            "Trained model could not be loaded from local filesystem or Railway Bucket. "
            f"Details: {exc}"
        ) from exc

    df = pd.DataFrame(rows)

    for column in ML_NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    for column in ML_CATEGORICAL_FEATURES:
        df[column] = df[column].fillna("Unknown").astype(str)

    feature_columns = ML_NUMERIC_FEATURES + ML_CATEGORICAL_FEATURES

    if isinstance(model, dict) and model.get("model_type") == "TwoStageZeroInflatedRandomForest":
        classifier = model["classifier"]
        regressor = model["regressor"]

        event_probabilities = classifier.predict_proba(df[feature_columns])[:, 1]
        expected_count_if_event = regressor.predict(df[feature_columns])
        calibration_factor = float(model.get("calibration_factor", 1.0) or 1.0)

        predicted_values = [
            max(0.0, float(probability)) * max(0.0, float(expected_count)) * calibration_factor
            for probability, expected_count in zip(event_probabilities, expected_count_if_event)
        ]

        model_type = "TwoStageZeroInflatedRandomForest"
    else:
        event_probabilities = [None] * len(df)
        expected_count_if_event = [None] * len(df)
        predicted_values = [max(0.0, float(value)) for value in model.predict(df[feature_columns])]
        model_type = type(model).__name__

    predictions: list[dict[str, Any]] = []

    for row, predicted, event_probability, count_if_event in zip(
        rows,
        predicted_values,
        event_probabilities,
        expected_count_if_event,
    ):
        predicted_value = max(0.0, float(predicted))
        feature_timestamp = row.get("feature_timestamp")
        forecast_for = feature_timestamp + timedelta(hours=1) if feature_timestamp else None

        probability_value = None if event_probability is None else max(0.0, min(1.0, float(event_probability)))
        count_if_event_value = None if count_if_event is None else max(0.0, float(count_if_event))

        predictions.append(
            {
                "feature_timestamp": feature_timestamp.isoformat() if feature_timestamp else None,
                "forecast_for": forecast_for.isoformat() if forecast_for else None,
                "police_district": row.get("police_district"),
                "incident_category": row.get("incident_category"),
                "event_probability_next_hour": round(probability_value, 4) if probability_value is not None else None,
                "expected_incidents_if_event": round(count_if_event_value, 4) if count_if_event_value is not None else None,
                "calibration_factor": (
                    round(float(model.get("calibration_factor", 1.0)), 8)
                    if isinstance(model, dict) and model.get("model_type") == "TwoStageZeroInflatedRandomForest"
                    else None
                ),
                "predicted_incidents_next_hour": round(predicted_value, 4),
            }
        )

    predictions.sort(key=lambda item: item["predicted_incidents_next_hour"], reverse=True)

    return {
        "row_count": len(predictions),
        "predictions": predictions,
        "total_predicted_incidents_next_hour": round(
            sum(item["predicted_incidents_next_hour"] for item in predictions),
            4,
        ),
        "model_source": model_source,
        "model_runtime_type": model_type,
    }


def normalize_district_name(value: Any) -> str:
    return str(value or "").strip().upper()


def title_district_name(value: str) -> str:
    return normalize_district_name(value).title()


def get_geojson_district_name(properties: dict[str, Any]) -> str:
    for key in (
        "district",
        "DISTRICT",
        "police_district",
        "POLICE_DISTRICT",
        "name",
        "NAME",
        "district_name",
        "DISTRICT_NAME",
    ):
        if properties.get(key):
            return normalize_district_name(properties.get(key))

    return ""


def load_police_districts_geojson() -> dict[str, Any]:
    if not POLICE_DISTRICTS_GEOJSON_PATH.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {POLICE_DISTRICTS_GEOJSON_PATH}")

    with POLICE_DISTRICTS_GEOJSON_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def classify_projected_volume(value: float) -> str:
    if value >= 0.75:
        return "high"

    if value >= 0.40:
        return "medium"

    if value > 0:
        return "low"

    return "none"


def build_volume_forecast_geojson(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    by_district: dict[str, dict[str, Any]] = {}

    for item in predictions:
        district_key = normalize_district_name(item.get("police_district"))

        if district_key not in VALID_MAP_DISTRICTS:
            continue

        entry = by_district.setdefault(
            district_key,
            {
                "police_district": title_district_name(district_key),
                "predicted_incidents_next_hour": 0.0,
                "event_probability_max": 0.0,
                "categories": [],
            },
        )

        predicted_value = float(item.get("predicted_incidents_next_hour") or 0)
        probability = float(item.get("event_probability_next_hour") or 0)

        entry["predicted_incidents_next_hour"] += predicted_value
        entry["event_probability_max"] = max(entry["event_probability_max"], probability)

        entry["categories"].append(
            {
                "incident_category": item.get("incident_category") or "Unknown",
                "predicted_incidents_next_hour": round(predicted_value, 4),
                "event_probability_next_hour": round(probability, 4),
            }
        )

    for entry in by_district.values():
        entry["predicted_incidents_next_hour"] = round(entry["predicted_incidents_next_hour"], 4)
        entry["event_probability_max"] = round(entry["event_probability_max"], 4)
        entry["risk_level"] = classify_projected_volume(entry["predicted_incidents_next_hour"])
        entry["categories"] = sorted(
            entry["categories"],
            key=lambda row: row["predicted_incidents_next_hour"],
            reverse=True,
        )[:5]

    source_geojson = load_police_districts_geojson()
    output_features = []

    for feature in source_geojson.get("features", []):
        properties = dict(feature.get("properties") or {})
        district_key = get_geojson_district_name(properties)

        if district_key not in VALID_MAP_DISTRICTS:
            continue

        forecast = by_district.get(
            district_key,
            {
                "police_district": title_district_name(district_key),
                "predicted_incidents_next_hour": 0.0,
                "event_probability_max": 0.0,
                "risk_level": "none",
                "categories": [],
            },
        )

        properties.update(
            {
                "police_district": title_district_name(district_key),
                "district_key": district_key,
                "predicted_incidents_next_hour": forecast["predicted_incidents_next_hour"],
                "event_probability_max": forecast["event_probability_max"],
                "risk_level": forecast["risk_level"],
                "top_categories": forecast["categories"],
            }
        )

        output_features.append(
            {
                "type": "Feature",
                "geometry": feature.get("geometry"),
                "properties": properties,
            }
        )

    return {
        "type": "FeatureCollection",
        "features": output_features,
        "mapped_districts": len(by_district),
        "total_predicted_incidents_next_hour": round(
            sum(item["predicted_incidents_next_hour"] for item in by_district.values()),
            4,
        ),
    }


# -----------------------
# Risk classifier utilities
# -----------------------

RISK_CLASSIFIER_MODEL_NAME = os.environ.get(
    "RISK_CLASSIFIER_MODEL_NAME",
    "risk_classifier_random_forest_v1",
)

RISK_MODEL_CACHE: dict[str, Any] = {
    "model": None,
    "source": None,
}

RISK_NUMERIC_FEATURES = [
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

RISK_CATEGORICAL_FEATURES = [
    "police_district",
    "incident_category",
    "day_of_week",
]

RISK_LEVEL_ORDER = ["Low", "Medium", "High", "Very High"]


def get_risk_model_path() -> Path:
    return MODEL_DIR / f"{RISK_CLASSIFIER_MODEL_NAME}.joblib"


def get_risk_model_artifact_keys() -> dict[str, str]:
    return {
        "model": f"{MODEL_S3_PREFIX.rstrip('/')}/{RISK_CLASSIFIER_MODEL_NAME}.joblib",
        "metrics": f"{MODEL_S3_PREFIX.rstrip('/')}/{RISK_CLASSIFIER_MODEL_NAME}_metrics.json",
    }


def read_risk_model_metrics_from_bucket() -> dict[str, Any] | None:
    keys = get_risk_model_artifact_keys()

    try:
        raw = download_bytes_from_model_bucket(keys["metrics"])
    except Exception:
        return None

    return json.loads(raw.decode("utf-8"))


def read_risk_model_metrics() -> dict[str, Any] | None:
    metrics_path = MODEL_DIR / f"{RISK_CLASSIFIER_MODEL_NAME}_metrics.json"

    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    if MODEL_BUCKET_NAME and MODEL_BUCKET_ENDPOINT_URL:
        return read_risk_model_metrics_from_bucket()

    return None


def load_risk_model_from_bucket() -> Any:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Missing ML dependency. Add joblib to requirements.txt.") from exc

    keys = get_risk_model_artifact_keys()
    raw = download_bytes_from_model_bucket(keys["model"])
    return joblib.load(io.BytesIO(raw))


def load_risk_model() -> tuple[Any, str]:
    cached_model = RISK_MODEL_CACHE.get("model")
    cached_source = RISK_MODEL_CACHE.get("source")

    if cached_model is not None:
        return cached_model, str(cached_source or "memory_cache")

    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Missing ML dependency. Add joblib to requirements.txt.") from exc

    model_path = get_risk_model_path()

    if model_path.exists():
        model = joblib.load(model_path)
        RISK_MODEL_CACHE["model"] = model
        RISK_MODEL_CACHE["source"] = str(model_path)
        return model, str(model_path)

    model = load_risk_model_from_bucket()
    RISK_MODEL_CACHE["model"] = model
    RISK_MODEL_CACHE["source"] = f"s3://{MODEL_BUCKET_NAME}/{get_risk_model_artifact_keys()['model']}"

    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
    except Exception:
        pass

    return model, str(RISK_MODEL_CACHE["source"])


def risk_level_from_score(score: float) -> str:
    if score >= 0.75:
        return "Very High"
    if score >= 0.55:
        return "High"
    if score >= 0.30:
        return "Medium"
    return "Low"


def risk_level_sort_value(level: Any) -> int:
    normalized = str(level or "").strip().title()
    if normalized == "Very High":
        return 4
    if normalized == "High":
        return 3
    if normalized == "Medium":
        return 2
    if normalized == "Low":
        return 1
    return 0


def predict_risk_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Missing ML dependency. Add pandas to requirements.txt.") from exc

    if not rows:
        return {
            "predictions": [],
            "row_count": 0,
            "model_source": None,
            "model_runtime_type": None,
        }

    try:
        model_bundle, model_source = load_risk_model()
    except Exception as exc:
        raise FileNotFoundError(
            "Trained risk classifier model could not be loaded from local filesystem or Railway Bucket. "
            f"Details: {exc}"
        ) from exc

    df = pd.DataFrame(rows)

    numeric_features = list(model_bundle.get("numeric_features") or RISK_NUMERIC_FEATURES)
    categorical_features = list(model_bundle.get("categorical_features") or RISK_CATEGORICAL_FEATURES)
    feature_columns = list(model_bundle.get("feature_columns") or (numeric_features + categorical_features))

    for column in numeric_features:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    for column in categorical_features:
        df[column] = df[column].fillna("Unknown").astype(str)

    missing_columns = [column for column in feature_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required risk model feature columns: {missing_columns}")

    classifier = model_bundle.get("classifier") if isinstance(model_bundle, dict) else None
    regressor = model_bundle.get("regressor") if isinstance(model_bundle, dict) else None

    if classifier is None or regressor is None:
        raise ValueError("Invalid risk model artifact. Expected classifier and regressor in model bundle.")

    X = df[feature_columns]
    predicted_level_ids = classifier.predict(X)
    predicted_scores = regressor.predict(X)

    level_order = list(model_bundle.get("risk_level_order") or RISK_LEVEL_ORDER)
    int_to_level_raw = model_bundle.get("int_to_risk_level") or {}
    int_to_level = {int(key): value for key, value in int_to_level_raw.items()} if isinstance(int_to_level_raw, dict) else {}

    probabilities = None
    try:
        probabilities = classifier.predict_proba(X)
        classifier_classes = [int(value) for value in classifier.classes_]
    except Exception:
        probabilities = None
        classifier_classes = []

    predictions: list[dict[str, Any]] = []

    for index, (row, level_id, score) in enumerate(zip(rows, predicted_level_ids, predicted_scores)):
        score_value = max(0.0, min(1.0, float(score)))
        level_id_int = int(level_id)
        level = int_to_level.get(level_id_int) or (
            level_order[level_id_int] if 0 <= level_id_int < len(level_order) else risk_level_from_score(score_value)
        )

        feature_timestamp = row.get("feature_timestamp")
        forecast_for = feature_timestamp + timedelta(hours=1) if feature_timestamp else None

        level_probability = None
        if probabilities is not None and level_id_int in classifier_classes:
            class_index = classifier_classes.index(level_id_int)
            level_probability = float(probabilities[index][class_index])

        predictions.append(
            {
                "feature_timestamp": feature_timestamp.isoformat() if feature_timestamp else None,
                "forecast_for": forecast_for.isoformat() if forecast_for else None,
                "police_district": row.get("police_district"),
                "incident_category": row.get("incident_category"),
                "risk_score": round(score_value, 4),
                "risk_level": str(level),
                "risk_level_probability": round(level_probability, 4) if level_probability is not None else None,
            }
        )

    predictions.sort(
        key=lambda item: (
            float(item.get("risk_score") or 0),
            risk_level_sort_value(item.get("risk_level")),
        ),
        reverse=True,
    )

    return {
        "row_count": len(predictions),
        "predictions": predictions,
        "model_source": model_source,
        "model_runtime_type": model_bundle.get("model_type") if isinstance(model_bundle, dict) else type(model_bundle).__name__,
    }


def classify_district_risk_level(score: float) -> str:
    return risk_level_from_score(score)


def build_risk_forecast_geojson(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    by_district: dict[str, dict[str, Any]] = {}

    for item in predictions:
        district_key = normalize_district_name(item.get("police_district"))

        if district_key not in VALID_MAP_DISTRICTS:
            continue

        entry = by_district.setdefault(
            district_key,
            {
                "police_district": title_district_name(district_key),
                "risk_score_sum": 0.0,
                "risk_score_max": 0.0,
                "risk_level_max": "Low",
                "risk_level_probability_max": 0.0,
                "categories": [],
            },
        )

        risk_score = float(item.get("risk_score") or 0)
        risk_level = item.get("risk_level") or classify_district_risk_level(risk_score)
        risk_level_probability = float(item.get("risk_level_probability") or 0)

        entry["risk_score_sum"] += risk_score
        entry["risk_score_max"] = max(entry["risk_score_max"], risk_score)
        entry["risk_level_probability_max"] = max(entry["risk_level_probability_max"], risk_level_probability)

        if risk_level_sort_value(risk_level) > risk_level_sort_value(entry["risk_level_max"]):
            entry["risk_level_max"] = str(risk_level)

        entry["categories"].append(
            {
                "incident_category": item.get("incident_category") or "Unknown",
                "risk_score": round(risk_score, 4),
                "risk_level": str(risk_level),
                "risk_level_probability": round(risk_level_probability, 4),
            }
        )

    for entry in by_district.values():
        category_count = max(1, len(entry["categories"]))
        entry["risk_score_avg"] = round(entry["risk_score_sum"] / category_count, 4)
        entry["risk_score_max"] = round(entry["risk_score_max"], 4)
        entry["risk_level_probability_max"] = round(entry["risk_level_probability_max"], 4)
        entry["risk_level"] = classify_district_risk_level(entry["risk_score_max"])
        entry["categories"] = sorted(
            entry["categories"],
            key=lambda row: (row["risk_score"], risk_level_sort_value(row["risk_level"])),
            reverse=True,
        )[:5]

    source_geojson = load_police_districts_geojson()
    output_features = []

    for feature in source_geojson.get("features", []):
        properties = dict(feature.get("properties") or {})
        district_key = get_geojson_district_name(properties)

        if district_key not in VALID_MAP_DISTRICTS:
            continue

        forecast = by_district.get(
            district_key,
            {
                "police_district": title_district_name(district_key),
                "risk_score_avg": 0.0,
                "risk_score_max": 0.0,
                "risk_level": "Low",
                "risk_level_max": "Low",
                "risk_level_probability_max": 0.0,
                "categories": [],
            },
        )

        properties.update(
            {
                "police_district": title_district_name(district_key),
                "district_key": district_key,
                "risk_score_avg": forecast["risk_score_avg"],
                "risk_score_max": forecast["risk_score_max"],
                "risk_level": forecast["risk_level"],
                "risk_level_max": forecast.get("risk_level_max", forecast["risk_level"]),
                "risk_level_probability_max": forecast["risk_level_probability_max"],
                "top_risk_categories": forecast["categories"],
            }
        )

        output_features.append(
            {
                "type": "Feature",
                "geometry": feature.get("geometry"),
                "properties": properties,
            }
        )

    return {
        "type": "FeatureCollection",
        "features": output_features,
        "mapped_districts": len(by_district),
        "max_risk_score": round(
            max((item["risk_score_max"] for item in by_district.values()), default=0.0),
            4,
        ),
        "avg_risk_score": round(
            sum(item["risk_score_avg"] for item in by_district.values()) / max(1, len(by_district)),
            4,
        ),
    }

