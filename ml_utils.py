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
    "risk_classifier_random_forest_v3",
)

RISK_MODEL_CACHE: dict[str, Any] = {
    "model": None,
    "source": None,
}

RISK_BASE_NUMERIC_FEATURES = [
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

RISK_DERIVED_NUMERIC_FEATURES = [
    "category_severity_score",
    "is_night_hour",
    "district_incidents_last_24h",
    "district_incidents_last_7d",
    "category_citywide_last_24h",
    "district_activity_share_24h",
    "category_activity_share_24h",
    "recent_pressure_score",
    "short_term_surge_score",
    "category_surge_score",
    "district_surge_score",
    "severity_pressure_interaction",
]

RISK_NUMERIC_FEATURES = RISK_BASE_NUMERIC_FEATURES + RISK_DERIVED_NUMERIC_FEATURES

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



def percentile_from_sorted(sorted_values: list[float], percentile: float) -> float:
    """
    Lightweight percentile helper without numpy.

    Uses nearest-rank style indexing, which is enough for dashboard calibration.
    Expected percentile values are in [0, 1], for example 0.50, 0.75, 0.90.
    """
    if not sorted_values:
        return 0.0

    safe_percentile = max(0.0, min(1.0, float(percentile)))
    index = round((len(sorted_values) - 1) * safe_percentile)
    index = max(0, min(index, len(sorted_values) - 1))
    return float(sorted_values[index])


def clean_risk_scores(scores: list[float]) -> list[float]:
    return sorted(
        max(0.0, min(1.0, float(score)))
        for score in scores
        if score is not None
    )


def build_risk_score_thresholds(scores: list[float]) -> dict[str, Any]:
    """
    Build dynamic thresholds from the current prediction batch.

    Important:
    risk_classifier_random_forest_v3 can produce compressed and repeated scores.
    If many districts share the same score, simple p50/p75/p90 thresholds can
    collapse to the same value. To avoid making every district Low, this function
    also calculates unique-score thresholds and exposes min/max/unique_count.
    """
    clean_scores = clean_risk_scores(scores)

    if not clean_scores:
        return {
            "medium": 0.0,
            "high": 0.0,
            "very_high": 0.0,
            "min": 0.0,
            "max": 0.0,
            "unique_count": 0,
        }

    unique_scores = sorted(set(clean_scores))

    if len(unique_scores) == 1:
        only_score = float(unique_scores[0])
        return {
            "medium": only_score,
            "high": only_score,
            "very_high": only_score,
            "min": only_score,
            "max": only_score,
            "unique_count": 1,
        }

    return {
        "medium": percentile_from_sorted(unique_scores, 0.50),
        "high": percentile_from_sorted(unique_scores, 0.75),
        "very_high": percentile_from_sorted(unique_scores, 0.90),
        "min": float(clean_scores[0]),
        "max": float(clean_scores[-1]),
        "unique_count": len(unique_scores),
    }


def risk_level_from_score(
    score: float,
    thresholds: dict[str, Any] | None = None,
) -> str:
    """
    Convert a risk score into a display level.

    If thresholds are provided, levels are relative to the current batch.
    If thresholds are not provided, fixed V2 fallback thresholds are used.
    """
    score_value = max(0.0, min(1.0, float(score or 0)))

    if thresholds:
        unique_count = int(thresholds.get("unique_count", 0) or 0)
        min_score = float(thresholds.get("min", 0.0) or 0.0)
        max_score = float(thresholds.get("max", 0.0) or 0.0)

        if unique_count <= 1 or abs(max_score - min_score) < 1e-12:
            return "Low"

        very_high = float(thresholds.get("very_high", max_score) or 0.0)
        high = float(thresholds.get("high", very_high) or 0.0)
        medium = float(thresholds.get("medium", high) or 0.0)

        if score_value >= very_high:
            return "Very High"
        if score_value >= high:
            return "High"
        if score_value >= medium:
            return "Medium"
        return "Low"

    # Fixed fallback calibrated for risk_classifier_random_forest_v3.
    if score_value >= 0.28:
        return "Very High"
    if score_value >= 0.20:
        return "High"
    if score_value >= 0.12:
        return "Medium"
    return "Low"


def district_risk_level_from_rank(
    score: float,
    all_scores: list[float],
) -> str:
    """
    Rank-aware district-level classification.

    This is stricter than raw thresholds and is designed for map display:
    - If every district has the same score, everything is Low.
    - If one district is clearly the highest, it becomes Very High.
    - If there are few unique values, the lower repeated value stays Low instead
      of being promoted to Medium only because percentiles collapsed.
    """
    clean_scores = clean_risk_scores(all_scores)
    unique_scores = sorted(set(clean_scores))

    if not unique_scores:
        return "Low"

    score_value = max(0.0, min(1.0, float(score or 0)))

    if len(unique_scores) == 1:
        return "Low"

    rank_from_top = unique_scores[::-1].index(score_value) + 1 if score_value in unique_scores else len(unique_scores)

    if len(unique_scores) == 2:
        return "Very High" if rank_from_top == 1 else "Low"

    if len(unique_scores) == 3:
        if rank_from_top == 1:
            return "Very High"
        if rank_from_top == 2:
            return "Medium"
        return "Low"

    percentile_position = (unique_scores.index(score_value) + 1) / len(unique_scores)

    if percentile_position >= 0.90:
        return "Very High"
    if percentile_position >= 0.75:
        return "High"
    if percentile_position >= 0.50:
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



def risk_category_severity_score(category: Any) -> float:
    normalized = str(category or "").strip().lower()

    very_high_keywords = ["homicide", "sex offense", "weapons", "weapon"]
    high_keywords = [
        "assault",
        "robbery",
        "burglary",
        "arson",
        "offences against the family",
        "offenses against the family",
    ]
    medium_keywords = [
        "motor vehicle theft",
        "larceny",
        "theft",
        "stolen property",
        "fraud",
        "malicious mischief",
        "vandalism",
        "forgery",
        "embezzlement",
    ]
    low_keywords = ["drug", "disorderly", "liquor", "gambling", "prostitution"]

    if any(keyword in normalized for keyword in very_high_keywords):
        return 1.0
    if any(keyword in normalized for keyword in high_keywords):
        return 0.82
    if any(keyword in normalized for keyword in medium_keywords):
        return 0.58
    if any(keyword in normalized for keyword in low_keywords):
        return 0.42
    return 0.35


def risk_is_night_hour(hour: Any) -> bool:
    try:
        h = int(hour)
    except (TypeError, ValueError):
        return False
    return h >= 22 or h <= 5


def _minmax_pandas_series(series: Any) -> Any:
    import pandas as pd

    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if abs(min_value - max_value) < 1e-12:
        return pd.Series([0.0] * len(numeric), index=numeric.index)
    return (numeric - min_value) / (max_value - min_value)


def add_risk_derived_features_to_df(df: Any) -> Any:
    import pandas as pd

    df = df.copy()

    for column in RISK_BASE_NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    for column in RISK_CATEGORICAL_FEATURES:
        df[column] = df[column].fillna("Unknown").astype(str)

    df["category_severity_score"] = df["incident_category"].map(risk_category_severity_score)
    df["is_night_hour"] = df["hour_of_day"].map(risk_is_night_hour).astype(int)

    df["district_incidents_last_24h"] = (
        df.groupby(["feature_timestamp", "police_district"])["incidents_last_24h"].transform("sum")
    )
    df["district_incidents_last_7d"] = (
        df.groupby(["feature_timestamp", "police_district"])["incidents_last_7d"].transform("sum")
    )
    df["category_citywide_last_24h"] = (
        df.groupby(["feature_timestamp", "incident_category"])["incidents_last_24h"].transform("sum")
    )

    city_incidents_last_24h = df.groupby("feature_timestamp")["incidents_last_24h"].transform("sum")
    district_total_last_24h = df["district_incidents_last_24h"].replace(0, pd.NA)
    city_total_last_24h = city_incidents_last_24h.replace(0, pd.NA)

    df["district_activity_share_24h"] = (
        df["district_incidents_last_24h"] / city_total_last_24h
    ).fillna(0.0)
    df["category_activity_share_24h"] = (
        df["incidents_last_24h"] / district_total_last_24h
    ).fillna(0.0)

    # V3: dynamic short-term pressure. 1h and 3h dominate; long-term history is only context.
    df["recent_pressure_score"] = (
        0.62 * _minmax_pandas_series(df["incidents_last_1h"])
        + 0.28 * _minmax_pandas_series(df["incidents_last_3h"])
        + 0.10 * _minmax_pandas_series(df["incidents_last_6h"])
    ).clip(0, 1)

    # Inference-safe surge features. These are calculated from the current feature batch.
    # They approximate "unusual for this district/category/hour" without needing extra DB queries.
    baseline_keys = ["police_district", "incident_category", "hour_of_day"]
    district_hour_keys = ["police_district", "hour_of_day"]
    category_hour_keys = ["incident_category", "hour_of_day"]

    baseline_3h = df.groupby(baseline_keys)["incidents_last_3h"].transform("mean").replace(0, pd.NA)
    baseline_24h = df.groupby(baseline_keys)["incidents_last_24h"].transform("mean").replace(0, pd.NA)
    district_baseline_24h = (
        df.groupby(district_hour_keys)["district_incidents_last_24h"].transform("mean").replace(0, pd.NA)
    )
    category_baseline_24h = (
        df.groupby(category_hour_keys)["category_citywide_last_24h"].transform("mean").replace(0, pd.NA)
    )

    category_surge_raw = (
        (df["incidents_last_3h"] / baseline_3h).fillna(0.0)
        + (df["incidents_last_24h"] / baseline_24h).fillna(0.0)
    ) / 2.0
    district_surge_raw = (
        df["district_incidents_last_24h"] / district_baseline_24h
    ).fillna(0.0)
    category_citywide_surge_raw = (
        df["category_citywide_last_24h"] / category_baseline_24h
    ).fillna(0.0)

    df["category_surge_score"] = _minmax_pandas_series(category_surge_raw.clip(0, 4))
    df["district_surge_score"] = _minmax_pandas_series(district_surge_raw.clip(0, 4))
    df["short_term_surge_score"] = (
        0.55 * df["category_surge_score"]
        + 0.30 * df["district_surge_score"]
        + 0.15 * _minmax_pandas_series(category_citywide_surge_raw.clip(0, 4))
    ).clip(0, 1)

    df["severity_pressure_interaction"] = (
        pd.to_numeric(df["category_severity_score"], errors="coerce").fillna(0.35)
        * (
            0.65 * df["recent_pressure_score"]
            + 0.35 * df["short_term_surge_score"]
        )
    ).clip(0, 1)

    return df


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
    df = add_risk_derived_features_to_df(df)

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

    score_values = [max(0.0, min(1.0, float(score))) for score in predicted_scores]
    score_thresholds = build_risk_score_thresholds(score_values)

    predictions: list[dict[str, Any]] = []

    for index, (row, level_id, score_value) in enumerate(zip(rows, predicted_level_ids, score_values)):
        level_id_int = int(level_id)

        # The regressor score is the source of truth for the visual/dashboard level.
        # The classifier probability is kept only as a confidence/debug signal.
        level = risk_level_from_score(score_value, score_thresholds)

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
        "risk_score_thresholds": {
            key: round(value, 4)
            for key, value in score_thresholds.items()
        },
        "level_strategy": "dynamic_score_percentiles",
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

        entry["categories"].append(
            {
                "incident_category": item.get("incident_category") or "Unknown",
                "risk_score": round(risk_score, 4),
                "risk_level_probability": round(risk_level_probability, 4),
            }
        )

    for entry in by_district.values():
        category_count = max(1, len(entry["categories"]))
        entry["risk_score_avg"] = round(entry["risk_score_sum"] / category_count, 4)
        entry["risk_score_max"] = round(entry["risk_score_max"], 4)
        entry["risk_level_probability_max"] = round(entry["risk_level_probability_max"], 4)

    district_scores = [entry["risk_score_max"] for entry in by_district.values()]
    district_score_thresholds = build_risk_score_thresholds(district_scores)

    for entry in by_district.values():
        entry["risk_level"] = district_risk_level_from_rank(
            entry["risk_score_max"],
            district_scores,
        )
        entry["categories"] = sorted(
            entry["categories"],
            key=lambda row: row["risk_score"],
            reverse=True,
        )[:5]

        for index, category in enumerate(entry["categories"], start=1):
            category["rank"] = index

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
        "risk_score_thresholds": {
            key: round(value, 4)
            for key, value in district_score_thresholds.items()
        },
        "level_strategy": "dynamic_district_score_percentiles",
        "level_source": "district_risk_score_max_rank",
        "category_level_policy": "top_risk_categories are ranked by risk_score only; they do not define district risk_level",
    }



# -----------------------
# Route risk model utilities
# -----------------------

ROUTE_RISK_MODEL_NAME = os.environ.get("ROUTE_RISK_MODEL_NAME", "ml_risk_route")
ROUTE_RISK_MODEL_CACHE: dict[str, Any] = {"model": None, "source": None}

ROUTE_RISK_NUMERIC_FEATURES = [
    "travel_hour",
    "incidents_near_route_100m_24h",
    "incidents_near_route_250m_24h",
    "incidents_near_route_500m_24h",
    "incidents_near_route_7d",
    "theft_ratio_near_route_7d",
    "assault_ratio_near_route_7d",
    "night_ratio_near_route_7d",
    "avg_distance_incidents_m",
    "max_segment_density",
]

ROUTE_RISK_CATEGORICAL_FEATURES = ["travel_day_of_week"]


def get_route_risk_model_path() -> Path:
    return MODEL_DIR / f"{ROUTE_RISK_MODEL_NAME}.joblib"


def get_route_risk_model_artifact_keys() -> dict[str, str]:
    return {
        "model": f"{MODEL_S3_PREFIX.rstrip('/')}/{ROUTE_RISK_MODEL_NAME}.joblib",
        "metrics": f"{MODEL_S3_PREFIX.rstrip('/')}/{ROUTE_RISK_MODEL_NAME}_metrics.json",
    }


def read_route_risk_model_metrics_from_bucket() -> dict[str, Any] | None:
    keys = get_route_risk_model_artifact_keys()
    try:
        raw = download_bytes_from_model_bucket(keys["metrics"])
    except Exception:
        return None
    return json.loads(raw.decode("utf-8"))


def read_route_risk_model_metrics() -> dict[str, Any] | None:
    metrics_path = MODEL_DIR / f"{ROUTE_RISK_MODEL_NAME}_metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    if MODEL_BUCKET_NAME and MODEL_BUCKET_ENDPOINT_URL:
        return read_route_risk_model_metrics_from_bucket()
    return None


def load_route_risk_model_from_bucket() -> Any:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Missing ML dependency. Add joblib to requirements.txt.") from exc

    keys = get_route_risk_model_artifact_keys()
    raw = download_bytes_from_model_bucket(keys["model"])
    return joblib.load(io.BytesIO(raw))


def load_route_risk_model() -> tuple[Any, str]:
    cached_model = ROUTE_RISK_MODEL_CACHE.get("model")
    cached_source = ROUTE_RISK_MODEL_CACHE.get("source")
    if cached_model is not None:
        return cached_model, str(cached_source or "memory_cache")

    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Missing ML dependency. Add joblib to requirements.txt.") from exc

    model_path = get_route_risk_model_path()
    if model_path.exists():
        model = joblib.load(model_path)
        ROUTE_RISK_MODEL_CACHE["model"] = model
        ROUTE_RISK_MODEL_CACHE["source"] = str(model_path)
        return model, str(model_path)

    model = load_route_risk_model_from_bucket()
    ROUTE_RISK_MODEL_CACHE["model"] = model
    ROUTE_RISK_MODEL_CACHE["source"] = f"s3://{MODEL_BUCKET_NAME}/{get_route_risk_model_artifact_keys()['model']}"

    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
    except Exception:
        pass

    return model, str(ROUTE_RISK_MODEL_CACHE["source"])


def decode_polyline(encoded: str) -> list[list[float]]:
    """Decode Google/OTP encoded polyline into [[lat, lon], ...]."""
    if not encoded:
        return []

    index = 0
    coordinates: list[list[float]] = []
    lat = 0
    lng = 0

    while index < len(encoded):
        result = 0
        shift = 0
        while True:
            byte = ord(encoded[index]) - 63
            index += 1
            result |= (byte & 0x1F) << shift
            shift += 5
            if byte < 0x20:
                break
        delta_lat = ~(result >> 1) if result & 1 else result >> 1
        lat += delta_lat

        result = 0
        shift = 0
        while True:
            byte = ord(encoded[index]) - 63
            index += 1
            result |= (byte & 0x1F) << shift
            shift += 5
            if byte < 0x20:
                break
        delta_lng = ~(result >> 1) if result & 1 else result >> 1
        lng += delta_lng

        coordinates.append([lat / 1e5, lng / 1e5])

    return coordinates


def safe_epoch_ms_to_datetime(value: Any) -> datetime:
    try:
        milliseconds = int(value)
        return datetime.fromtimestamp(milliseconds / 1000.0)
    except Exception:
        return datetime.utcnow()


def normalize_route_points(points: list[list[float]]) -> list[list[float]]:
    cleaned: list[list[float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            lat = float(point[0])
            lon = float(point[1])
        except (TypeError, ValueError):
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        if cleaned and abs(cleaned[-1][0] - lat) < 1e-8 and abs(cleaned[-1][1] - lon) < 1e-8:
            continue
        cleaned.append([lat, lon])
    return cleaned


def build_linestring_wkt(points: list[list[float]]) -> str:
    clean_points = normalize_route_points(points)
    if len(clean_points) < 2:
        raise ValueError("A route needs at least two valid coordinates to build a LINESTRING.")
    lon_lat_pairs = [f"{lon:.7f} {lat:.7f}" for lat, lon in clean_points]
    return "LINESTRING(" + ", ".join(lon_lat_pairs) + ")"


def extract_itineraries_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("itineraries"), list):
        return payload["itineraries"]

    collapse_keys = sorted(key for key in payload.keys() if str(key).startswith("collapse"))
    if collapse_keys:
        return [payload[key] | {"itinerary_id": key} for key in collapse_keys if isinstance(payload.get(key), dict)]

    if isinstance(payload.get("legs"), list):
        return [payload]

    raise ValueError("Payload must include 'itineraries', collapse* route objects, or a single object with 'legs'.")


def decode_itinerary_points(itinerary: dict[str, Any]) -> tuple[list[list[float]], list[dict[str, Any]]]:
    all_points: list[list[float]] = []
    decoded_legs: list[dict[str, Any]] = []

    for leg_index, leg in enumerate(itinerary.get("legs") or []):
        encoded = ((leg.get("legGeometry") or {}).get("points") or "").strip()
        points = normalize_route_points(decode_polyline(encoded)) if encoded else []
        if not points:
            from_point = leg.get("from") or {}
            to_point = leg.get("to") or {}
            points = normalize_route_points(
                [
                    [from_point.get("lat"), from_point.get("lon")],
                    [to_point.get("lat"), to_point.get("lon")],
                ]
            )

        if all_points and points:
            all_points.extend(points[1:] if all_points[-1] == points[0] else points)
        else:
            all_points.extend(points)

        decoded_legs.append(
            {
                "leg_index": leg_index,
                "mode": leg.get("mode"),
                "route": leg.get("route"),
                "agency": leg.get("agency"),
                "from": leg.get("from"),
                "to": leg.get("to"),
                "startTime": leg.get("startTime"),
                "endTime": leg.get("endTime"),
                "duration": leg.get("duration"),
                "decoded_point_count": len(points),
                "decoded_points": points,
            }
        )

    return normalize_route_points(all_points), decoded_legs


def route_risk_level_from_score(score: float) -> str:
    value = max(0.0, min(1.0, float(score or 0)))
    if value >= 0.75:
        return "Very High"
    if value >= 0.55:
        return "High"
    if value >= 0.30:
        return "Medium"
    return "Low"


def insert_route_request(
    route_wkt: str,
    origin_lat: float | None,
    origin_lon: float | None,
    dest_lat: float | None,
    dest_lon: float | None,
    travel_hour: int,
    travel_day_of_week: str,
) -> int:
    rows = fetch_all_dict(
        """
        INSERT INTO route_requests (
            requested_at,
            origin_lat,
            origin_lon,
            dest_lat,
            dest_lon,
            route_geom,
            travel_hour,
            travel_day_of_week
        )
        VALUES (
            NOW(), %s, %s, %s, %s, ST_GeomFromText(%s, 4326), %s, %s
        )
        RETURNING route_id;
        """,
        (origin_lat, origin_lon, dest_lat, dest_lon, route_wkt, travel_hour, travel_day_of_week),
    )
    return int(rows[0]["route_id"])


def compute_route_risk_features_from_db(
    route_wkt: str,
    travel_hour: int,
    travel_day_of_week: str,
) -> dict[str, Any]:
    rows = fetch_all_dict(
        """
        WITH route AS (
            SELECT ST_GeomFromText(%s, 4326) AS geom
        ), incidents_24h AS (
            SELECT i.*
            FROM incidents_raw i, route r
            WHERE i.geom IS NOT NULL
              AND i.incident_datetime >= NOW() - INTERVAL '24 hours'
              AND ST_DWithin(i.geom::geography, r.geom::geography, 500)
        ), incidents_7d AS (
            SELECT i.*
            FROM incidents_raw i, route r
            WHERE i.geom IS NOT NULL
              AND i.incident_datetime >= NOW() - INTERVAL '7 days'
              AND ST_DWithin(i.geom::geography, r.geom::geography, 500)
        ), segments AS (
            SELECT ST_LineSubstring(r.geom, gs::float8, LEAST((gs + 0.05)::float8, 1.0)) AS geom
            FROM route r, generate_series(0.0, 0.95, 0.05) AS gs
        ), segment_counts AS (
            SELECT COUNT(i.row_id) AS incident_count
            FROM segments s
            LEFT JOIN incidents_raw i
              ON i.geom IS NOT NULL
             AND i.incident_datetime >= NOW() - INTERVAL '24 hours'
             AND ST_DWithin(i.geom::geography, s.geom::geography, 250)
            GROUP BY s.geom
        )
        SELECT
            %s::int AS travel_hour,
            %s::text AS travel_day_of_week,
            (SELECT COUNT(*) FROM incidents_24h i, route r WHERE ST_DWithin(i.geom::geography, r.geom::geography, 100))::int AS incidents_near_route_100m_24h,
            (SELECT COUNT(*) FROM incidents_24h i, route r WHERE ST_DWithin(i.geom::geography, r.geom::geography, 250))::int AS incidents_near_route_250m_24h,
            (SELECT COUNT(*) FROM incidents_24h)::int AS incidents_near_route_500m_24h,
            (SELECT COUNT(*) FROM incidents_7d)::int AS incidents_near_route_7d,
            COALESCE((SELECT AVG(CASE WHEN incident_category ILIKE '%%theft%%' OR incident_category ILIKE '%%larceny%%' THEN 1.0 ELSE 0.0 END) FROM incidents_7d), 0.0) AS theft_ratio_near_route_7d,
            COALESCE((SELECT AVG(CASE WHEN incident_category ILIKE '%%assault%%' THEN 1.0 ELSE 0.0 END) FROM incidents_7d), 0.0) AS assault_ratio_near_route_7d,
            COALESCE((SELECT AVG(CASE WHEN EXTRACT(HOUR FROM incident_datetime) >= 20 OR EXTRACT(HOUR FROM incident_datetime) <= 5 THEN 1.0 ELSE 0.0 END) FROM incidents_7d), 0.0) AS night_ratio_near_route_7d,
            COALESCE((SELECT AVG(ST_Distance(i.geom::geography, r.geom::geography)) FROM incidents_7d i, route r), 9999.0) AS avg_distance_incidents_m,
            COALESCE((SELECT MAX(incident_count)::double precision FROM segment_counts), 0.0) AS max_segment_density;
        """,
        (route_wkt, travel_hour, travel_day_of_week),
    )

    if not rows:
        raise RuntimeError("Could not compute route risk features.")
    return dict(rows[0])


def insert_route_risk_feature(route_id: int, features: dict[str, Any]) -> int:
    """Insert route ML features with a bootstrap target.

    Why target is filled here:
    - route_risk_features is the training table for ml_risk_route.
    - target_risk_score / target_risk_level must represent the label that the
      supervised model learns from.
    - Until there is real post-trip feedback, we bootstrap that label with the
      same deterministic risk formula used by the fallback predictor.

    Later, this target can be replaced by a stronger label, for example
    incidents near the route in the hour after the route was requested.
    """
    target_risk_score = round(float(heuristic_route_risk_score(features)), 6)
    target_risk_level = route_risk_level_from_score(target_risk_score)

    rows = fetch_all_dict(
        """
        INSERT INTO route_risk_features (
            route_id,
            computed_at,
            travel_hour,
            travel_day_of_week,
            incidents_near_route_100m_24h,
            incidents_near_route_250m_24h,
            incidents_near_route_500m_24h,
            incidents_near_route_7d,
            theft_ratio_near_route_7d,
            assault_ratio_near_route_7d,
            night_ratio_near_route_7d,
            avg_distance_incidents_m,
            max_segment_density,
            target_risk_score,
            target_risk_level
        )
        VALUES (%s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING route_feature_id;
        """,
        (
            route_id,
            features.get("travel_hour"),
            features.get("travel_day_of_week"),
            features.get("incidents_near_route_100m_24h"),
            features.get("incidents_near_route_250m_24h"),
            features.get("incidents_near_route_500m_24h"),
            features.get("incidents_near_route_7d"),
            features.get("theft_ratio_near_route_7d"),
            features.get("assault_ratio_near_route_7d"),
            features.get("night_ratio_near_route_7d"),
            features.get("avg_distance_incidents_m"),
            features.get("max_segment_density"),
            target_risk_score,
            target_risk_level,
        ),
    )
    return int(rows[0]["route_feature_id"])


def insert_route_risk_prediction(route_id: int, score: float, level: str) -> int:
    rows = fetch_all_dict(
        """
        INSERT INTO route_risk_predictions (
            model_name,
            generated_at,
            route_id,
            risk_score,
            risk_level
        )
        VALUES (%s, NOW(), %s, %s, %s)
        RETURNING prediction_id;
        """,
        (ROUTE_RISK_MODEL_NAME, route_id, score, level),
    )
    return int(rows[0]["prediction_id"])


def heuristic_route_risk_score(features: dict[str, Any]) -> float:
    count_250 = min(float(features.get("incidents_near_route_250m_24h") or 0) / 20.0, 1.0)
    count_7d = min(float(features.get("incidents_near_route_7d") or 0) / 120.0, 1.0)
    max_segment = min(float(features.get("max_segment_density") or 0) / 12.0, 1.0)
    avg_distance = float(features.get("avg_distance_incidents_m") or 9999)
    distance_pressure = 1.0 - min(avg_distance / 500.0, 1.0)
    night = max(0.0, min(1.0, float(features.get("night_ratio_near_route_7d") or 0)))
    theft = max(0.0, min(1.0, float(features.get("theft_ratio_near_route_7d") or 0)))
    assault = max(0.0, min(1.0, float(features.get("assault_ratio_near_route_7d") or 0)))

    score = (
        0.25 * count_250
        + 0.22 * count_7d
        + 0.20 * max_segment
        + 0.13 * distance_pressure
        + 0.08 * night
        + 0.06 * theft
        + 0.06 * assault
    )
    return max(0.0, min(1.0, float(score)))




def backfill_route_risk_feature_targets(limit: int | None = None) -> dict[str, Any]:
    """Backfill target_risk_score and target_risk_level for existing NULL rows.

    Use this once after deploying this version if old rows were inserted before
    targets were calculated. It updates only rows where either target is NULL.
    """
    limit_clause = ""
    params: list[Any] = []

    if limit is not None:
        limit_clause = "LIMIT %s"
        params.append(max(1, int(limit)))

    rows = fetch_all_dict(
        f"""
        SELECT
            route_feature_id,
            travel_hour,
            travel_day_of_week,
            incidents_near_route_100m_24h,
            incidents_near_route_250m_24h,
            incidents_near_route_500m_24h,
            incidents_near_route_7d,
            theft_ratio_near_route_7d,
            assault_ratio_near_route_7d,
            night_ratio_near_route_7d,
            avg_distance_incidents_m,
            max_segment_density
        FROM route_risk_features
        WHERE target_risk_score IS NULL
           OR target_risk_level IS NULL
        ORDER BY route_feature_id ASC
        {limit_clause};
        """,
        tuple(params),
    )

    updated = 0
    for row in rows:
        score = round(float(heuristic_route_risk_score(row)), 6)
        level = route_risk_level_from_score(score)
        fetch_all_dict(
            """
            UPDATE route_risk_features
            SET target_risk_score = %s,
                target_risk_level = %s,
                updated_at = NOW()
            WHERE route_feature_id = %s;
            """,
            (score, level, row["route_feature_id"]),
        )
        updated += 1

    remaining_rows = fetch_all_dict(
        """
        SELECT COUNT(*) AS count
        FROM route_risk_features
        WHERE target_risk_score IS NULL
           OR target_risk_level IS NULL;
        """
    )

    return {
        "status": "ok",
        "updated_rows": updated,
        "remaining_null_rows": int(remaining_rows[0]["count"]),
    }


def predict_route_risk_from_features(features: dict[str, Any]) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Missing ML dependency. Add pandas to requirements.txt.") from exc

    model_source = None
    model_runtime_type = None

    try:
        model, model_source = load_route_risk_model()
        df = pd.DataFrame([features])
        for column in ROUTE_RISK_NUMERIC_FEATURES:
            df[column] = pd.to_numeric(df.get(column), errors="coerce").fillna(0)
        df["avg_distance_incidents_m"] = pd.to_numeric(df.get("avg_distance_incidents_m"), errors="coerce").fillna(9999)
        for column in ROUTE_RISK_CATEGORICAL_FEATURES:
            df[column] = df.get(column, "Unknown").fillna("Unknown").astype(str)

        if isinstance(model, dict) and model.get("model_type") == "RouteRiskRandomForestRegressor":
            pipeline = model["pipeline"]
            feature_columns = model.get("feature_columns") or ROUTE_RISK_NUMERIC_FEATURES + ROUTE_RISK_CATEGORICAL_FEATURES
            score = float(pipeline.predict(df[feature_columns])[0])
            model_runtime_type = model.get("model_type")
        else:
            score = float(model.predict(df[ROUTE_RISK_NUMERIC_FEATURES + ROUTE_RISK_CATEGORICAL_FEATURES])[0])
            model_runtime_type = type(model).__name__

        score = max(0.0, min(1.0, score))
        fallback_used = False
    except Exception as exc:
        score = heuristic_route_risk_score(features)
        model_source = f"heuristic_fallback: {exc}"
        model_runtime_type = "heuristic_route_risk_score"
        fallback_used = True

    level = route_risk_level_from_score(score)
    return {
        "risk_score": round(score, 4),
        "risk_level": level,
        "model_name": ROUTE_RISK_MODEL_NAME,
        "model_source": model_source,
        "model_runtime_type": model_runtime_type,
        "fallback_used": fallback_used,
    }


def evaluate_route_itinerary_risk(
    itinerary: dict[str, Any],
    origin_lat: float | None = None,
    origin_lon: float | None = None,
    dest_lat: float | None = None,
    dest_lon: float | None = None,
    save: bool = True,
) -> dict[str, Any]:
    points, decoded_legs = decode_itinerary_points(itinerary)
    route_wkt = build_linestring_wkt(points)

    start_time = itinerary.get("startTime")
    if not start_time and itinerary.get("legs"):
        start_time = (itinerary["legs"][0] or {}).get("startTime")
    travel_dt = safe_epoch_ms_to_datetime(start_time)
    travel_hour = int(travel_dt.hour)
    travel_day_of_week = travel_dt.strftime("%A")

    if origin_lat is None or origin_lon is None:
        if points:
            origin_lat, origin_lon = points[0]
    if dest_lat is None or dest_lon is None:
        if points:
            dest_lat, dest_lon = points[-1]

    route_id = insert_route_request(
        route_wkt=route_wkt,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        dest_lat=dest_lat,
        dest_lon=dest_lon,
        travel_hour=travel_hour,
        travel_day_of_week=travel_day_of_week,
    ) if save else None

    features = compute_route_risk_features_from_db(route_wkt, travel_hour, travel_day_of_week)
    route_feature_id = insert_route_risk_feature(route_id, features) if save and route_id is not None else None
    prediction = predict_route_risk_from_features(features)
    prediction_id = insert_route_risk_prediction(route_id, prediction["risk_score"], prediction["risk_level"]) if save and route_id is not None else None

    return {
        "route_id": route_id,
        "route_feature_id": route_feature_id,
        "prediction_id": prediction_id,
        "risk_score": prediction["risk_score"],
        "risk_level": prediction["risk_level"],
        "model_name": prediction["model_name"],
        "model_source": prediction["model_source"],
        "model_runtime_type": prediction["model_runtime_type"],
        "fallback_used": prediction["fallback_used"],
        "features": features,
        "decoded_point_count": len(points),
        "decoded_legs": decoded_legs,
    }


def evaluate_itineraries_route_risk(payload: dict[str, Any]) -> dict[str, Any]:
    itineraries = extract_itineraries_from_payload(payload)
    save = bool(payload.get("save", True))
    origin_lat = payload.get("origin_lat")
    origin_lon = payload.get("origin_lon")
    dest_lat = payload.get("dest_lat")
    dest_lon = payload.get("dest_lon")

    results: list[dict[str, Any]] = []
    for index, itinerary in enumerate(itineraries):
        item = evaluate_route_itinerary_risk(
            itinerary,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            dest_lat=dest_lat,
            dest_lon=dest_lon,
            save=save,
        )
        item["itinerary_index"] = index
        item["itinerary_id"] = itinerary.get("itinerary_id") or itinerary.get("id") or f"itinerary_{index + 1}"
        item["duration"] = itinerary.get("duration")
        item["generalizedCost"] = itinerary.get("generalizedCost")
        item["walkDistance"] = itinerary.get("walkDistance")
        results.append(item)

    sorted_results = sorted(results, key=lambda row: float(row.get("risk_score") or 0))
    return {
        "route_count": len(results),
        "routes": results,
        "safest_route": sorted_results[0] if sorted_results else None,
        "highest_risk_route": sorted_results[-1] if sorted_results else None,
    }
