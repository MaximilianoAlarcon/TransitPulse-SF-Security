"""
Hotspot clustering utilities for CI San Francisco.

This module is intended to live next to app.py and be imported by app.py.
It does not train or persist a model. It reads recent incidents, applies DBSCAN,
and returns a GeoJSON FeatureCollection ready for Leaflet.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Callable

try:
    import numpy as np
    from sklearn.cluster import DBSCAN
except ImportError as exc:  # fail with a clear app-level error if dependency is missing
    np = None  # type: ignore[assignment]
    DBSCAN = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from psycopg2.extras import RealDictCursor

EARTH_RADIUS_METERS = 6_371_000.0
DEFAULT_EPS_METERS = 250.0
DEFAULT_MIN_SAMPLES = 8
DEFAULT_MAX_POINTS = 12_000
DEFAULT_MAX_CLUSTERS = 250
DEFAULT_CIRCLE_STEPS = 32


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _parse_positive_float(value: Any, default: float, min_value: float, max_value: float) -> float:
    parsed = _safe_float(value, default)
    return max(min_value, min(max_value, parsed))


def _parse_positive_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def _circle_polygon(lon: float, lat: float, radius_meters: float, steps: int = DEFAULT_CIRCLE_STEPS) -> list[list[float]]:
    """Return an approximate WGS84 circle polygon as [lon, lat] coordinates."""
    coords: list[list[float]] = []
    lat_rad = math.radians(lat)
    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = max(1.0, 111_320.0 * math.cos(lat_rad))

    for i in range(steps + 1):
        angle = (2.0 * math.pi * i) / steps
        dx = math.cos(angle) * radius_meters
        dy = math.sin(angle) * radius_meters
        coords.append([lon + (dx / meters_per_degree_lon), lat + (dy / meters_per_degree_lat)])

    return coords


def _haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2.0 * EARTH_RADIUS_METERS * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def fetch_cluster_input_rows(
    get_db_connection: Callable[[], Any],
    *,
    start_dt: datetime,
    end_dt: datetime,
    category_filter_values: list[str],
    district: str = "all",
    category: str = "all",
    max_points: int = DEFAULT_MAX_POINTS,
) -> list[dict[str, Any]]:
    """Fetch incident points restricted to relevant product categories and UI filters."""
    if not category_filter_values:
        return []

    conditions = [
        "r.incident_datetime >= %s",
        "r.incident_datetime < %s",
        "r.latitude IS NOT NULL",
        "r.longitude IS NOT NULL",
        "r.latitude BETWEEN 37.60 AND 37.90",
        "r.longitude BETWEEN -122.60 AND -122.30",
    ]
    params: list[Any] = [start_dt, end_dt]

    allowed_categories = list(dict.fromkeys(category_filter_values))

    if category and category.lower() != "all":
        if category not in allowed_categories:
            return []
        conditions.append("r.incident_category = %s")
        params.append(category)
    else:
        placeholders = ", ".join(["%s"] * len(allowed_categories))
        conditions.append(f"r.incident_category IN ({placeholders})")
        params.extend(allowed_categories)

    if district and district.lower() != "all":
        conditions.append("r.police_district = %s")
        params.append(district)

    params.append(max_points)

    query = f"""
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
        WHERE {' AND '.join(conditions)}
        ORDER BY r.incident_datetime DESC
        LIMIT %s;
    """

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, tuple(params))
            return [dict(row) for row in cur.fetchall()]


def _cluster_group(
    rows: list[dict[str, Any]],
    *,
    eps_meters: float,
    min_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if _IMPORT_ERROR is not None or np is None or DBSCAN is None:
        raise RuntimeError("Missing clustering dependencies. Add scikit-learn and numpy to requirements.txt.") from _IMPORT_ERROR

    if len(rows) < min_samples:
        return [], rows

    coordinates_radians = np.radians(
        [[_safe_float(row["latitude"]), _safe_float(row["longitude"])] for row in rows]
    )
    eps_radians = eps_meters / EARTH_RADIUS_METERS

    labels = DBSCAN(
        eps=eps_radians,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    ).fit_predict(coordinates_radians)

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    noise_rows: list[dict[str, Any]] = []

    for row, raw_label in zip(rows, labels):
        label = int(raw_label)
        if label == -1:
            noise_rows.append(row)
            continue
        grouped[label].append(row)

    clusters: list[dict[str, Any]] = []
    largest_cluster_size = max((len(items) for items in grouped.values()), default=1)

    for label, items in grouped.items():
        lats = [_safe_float(item["latitude"]) for item in items]
        lons = [_safe_float(item["longitude"]) for item in items]
        centroid_lat = sum(lats) / len(lats)
        centroid_lon = sum(lons) / len(lons)

        distances = [
            _haversine_distance_m(centroid_lat, centroid_lon, lat, lon)
            for lat, lon in zip(lats, lons)
        ]
        max_distance = max(distances, default=eps_meters)
        avg_distance = sum(distances) / len(distances) if distances else 0.0
        radius_meters = max(eps_meters, min(max_distance + eps_meters, eps_meters * 3.0))
        density_score = min(1.0, len(items) / max(float(largest_cluster_size), 1.0))

        district_counts = Counter(item.get("police_district") or "Unknown" for item in items)
        subcategory_counts = Counter(item.get("incident_subcategory") or "Unknown" for item in items)

        clusters.append(
            {
                "cluster_label": label,
                "point_count": len(items),
                "centroid_lat": centroid_lat,
                "centroid_lon": centroid_lon,
                "radius_meters": radius_meters,
                "avg_distance_meters": avg_distance,
                "density_score": density_score,
                "top_districts": [
                    {"district": district, "count": count}
                    for district, count in district_counts.most_common(3)
                ],
                "top_subcategories": [
                    {"subcategory": subcategory, "count": count}
                    for subcategory, count in subcategory_counts.most_common(3)
                ],
                "latest_incident_datetime": max(
                    (item.get("incident_datetime") for item in items if item.get("incident_datetime")),
                    default=None,
                ),
            }
        )

    clusters.sort(key=lambda item: (item["point_count"], item["density_score"]), reverse=True)
    return clusters, noise_rows


def build_hotspot_geojson(
    get_db_connection: Callable[[], Any],
    *,
    start_dt: datetime,
    end_dt: datetime,
    category_filter_values: list[str],
    district: str = "all",
    category: str = "all",
    eps_meters: Any = DEFAULT_EPS_METERS,
    min_samples: Any = DEFAULT_MIN_SAMPLES,
    max_points: Any = DEFAULT_MAX_POINTS,
    max_clusters: Any = DEFAULT_MAX_CLUSTERS,
) -> dict[str, Any]:
    """
    Return hotspot clusters as GeoJSON polygons.

    Design choice:
    - If category=all, cluster separately per incident_category so semantic categories do not get mixed.
    - If category is selected, cluster only that category.
    """
    safe_eps = _parse_positive_float(eps_meters, DEFAULT_EPS_METERS, 50.0, 2_000.0)
    safe_min_samples = _parse_positive_int(min_samples, DEFAULT_MIN_SAMPLES, 3, 100)
    safe_max_points = _parse_positive_int(max_points, DEFAULT_MAX_POINTS, 100, 75_000)
    safe_max_clusters = _parse_positive_int(max_clusters, DEFAULT_MAX_CLUSTERS, 1, 1_000)

    rows = fetch_cluster_input_rows(
        get_db_connection,
        start_dt=start_dt,
        end_dt=end_dt,
        category_filter_values=category_filter_values,
        district=district,
        category=category,
        max_points=safe_max_points,
    )

    rows_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_category[row.get("incident_category") or "Unknown"].append(row)

    features: list[dict[str, Any]] = []
    noise_features: list[dict[str, Any]] = []

    for incident_category, category_rows in sorted(rows_by_category.items()):
        clusters, noise_rows = _cluster_group(
            category_rows,
            eps_meters=safe_eps,
            min_samples=safe_min_samples,
        )

        for noise_index, noise_row in enumerate(noise_rows):
            incident_dt = noise_row.get("incident_datetime")
            noise_id = f"noise:{incident_category}:{noise_row.get('row_id') or noise_index}"
            noise_features.append(
                {
                    "type": "Feature",
                    "id": noise_id,
                    "geometry": {
                        "type": "Point",
                        "coordinates": [
                            round(_safe_float(noise_row.get("longitude")), 6),
                            round(_safe_float(noise_row.get("latitude")), 6),
                        ],
                    },
                    "properties": {
                        "hotspot_id": noise_id,
                        "incident_id": noise_row.get("row_id"),
                        "incident_category": incident_category,
                        "incident_subcategory": noise_row.get("incident_subcategory") or "Unknown",
                        "incident_description": noise_row.get("incident_description") or "",
                        "police_district": noise_row.get("police_district") or "Unknown",
                        "resolution": noise_row.get("resolution") or "Unknown",
                        "is_noise": True,
                        "is_hotspot": False,
                        "incident_datetime": incident_dt.isoformat() if incident_dt else None,
                        "time_window_start": start_dt.isoformat(),
                        "time_window_end": end_dt.isoformat(),
                    },
                }
            )

        for cluster in clusters:
            hotspot_id = f"{incident_category}:{cluster['cluster_label']}"
            latest_dt = cluster.get("latest_incident_datetime")
            features.append(
                {
                    "type": "Feature",
                    "id": hotspot_id,
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            _circle_polygon(
                                cluster["centroid_lon"],
                                cluster["centroid_lat"],
                                cluster["radius_meters"],
                            )
                        ],
                    },
                    "properties": {
                        "hotspot_id": hotspot_id,
                        "incident_category": incident_category,
                        "cluster_label": cluster["cluster_label"],
                        "point_count": cluster["point_count"],
                        "density_score": round(cluster["density_score"], 4),
                        "is_hotspot": cluster["point_count"] >= safe_min_samples,
                        "centroid": {
                            "lat": round(cluster["centroid_lat"], 6),
                            "lon": round(cluster["centroid_lon"], 6),
                        },
                        "radius_meters": round(cluster["radius_meters"], 2),
                        "avg_distance_meters": round(cluster["avg_distance_meters"], 2),
                        "top_districts": cluster["top_districts"],
                        "top_subcategories": cluster["top_subcategories"],
                        "latest_incident_datetime": latest_dt.isoformat() if latest_dt else None,
                        "time_window_start": start_dt.isoformat(),
                        "time_window_end": end_dt.isoformat(),
                    },
                }
            )

    features.sort(
        key=lambda feature: (
            feature["properties"]["point_count"],
            feature["properties"]["density_score"],
        ),
        reverse=True,
    )
    features = features[:safe_max_clusters]

    return {
        "status": "ok",
        "type": "FeatureCollection",
        "features": features,
        "summary": {
            "source_points": len(rows),
            "cluster_count": len(features),
            "noise_points": len(noise_features),
            "isolated_points_visible": True,
            "eps_meters": safe_eps,
            "min_samples": safe_min_samples,
            "max_points": safe_max_points,
            "max_clusters": safe_max_clusters,
            "clustered_by": "incident_category",
        },
        "noise_features": noise_features,
        "filters": {
            "district": district or "all",
            "category": category or "all",
            "time_window_start": start_dt.isoformat(),
            "time_window_end": end_dt.isoformat(),
            "category_filter_count": len(category_filter_values),
        },
    }
