BEGIN;

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS incidents_raw (
    row_id TEXT PRIMARY KEY,
    incident_datetime TIMESTAMP,
    incident_date DATE,
    incident_time TIME,
    incident_year INT,
    incident_day_of_week TEXT,
    report_datetime TIMESTAMP,
    incident_id TEXT,
    incident_number TEXT,
    report_type_code TEXT,
    report_type_description TEXT,
    filed_online BOOLEAN,
    incident_code TEXT,
    incident_category TEXT,
    incident_subcategory TEXT,
    incident_description TEXT,
    resolution TEXT,
    police_district TEXT,
    data_as_of TIMESTAMP,
    data_loaded_at TIMESTAMP,
    incident_hour INT,
    report_delay_minutes INT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    geom geometry(Point, 4326),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_incidents_raw_incident_datetime
    ON incidents_raw (incident_datetime);
CREATE INDEX IF NOT EXISTS idx_incidents_raw_incident_date
    ON incidents_raw (incident_date);
CREATE INDEX IF NOT EXISTS idx_incidents_raw_police_district
    ON incidents_raw (police_district);
CREATE INDEX IF NOT EXISTS idx_incidents_raw_incident_category
    ON incidents_raw (incident_category);
CREATE INDEX IF NOT EXISTS idx_incidents_raw_subcategory
    ON incidents_raw (incident_subcategory);
CREATE INDEX IF NOT EXISTS idx_incidents_raw_resolution
    ON incidents_raw (resolution);
CREATE INDEX IF NOT EXISTS idx_incidents_raw_geom
    ON incidents_raw USING GIST (geom);

CREATE TABLE IF NOT EXISTS incident_counts_hourly (
    bucket_start TIMESTAMP NOT NULL,
    police_district TEXT NOT NULL,
    incident_category TEXT NOT NULL,
    incident_subcategory TEXT,
    total_incidents INT NOT NULL,
    open_active_count INT NOT NULL,
    filed_online_count INT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (bucket_start, police_district, incident_category, incident_subcategory)
);

CREATE INDEX IF NOT EXISTS idx_incident_counts_hourly_bucket_start
    ON incident_counts_hourly (bucket_start);
CREATE INDEX IF NOT EXISTS idx_incident_counts_hourly_district_category
    ON incident_counts_hourly (police_district, incident_category);

CREATE TABLE IF NOT EXISTS incident_counts_daily (
    bucket_date DATE NOT NULL,
    police_district TEXT NOT NULL,
    incident_category TEXT NOT NULL,
    incident_subcategory TEXT,
    total_incidents INT NOT NULL,
    open_active_count INT NOT NULL,
    filed_online_count INT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (bucket_date, police_district, incident_category, incident_subcategory)
);

CREATE INDEX IF NOT EXISTS idx_incident_counts_daily_bucket_date
    ON incident_counts_daily (bucket_date);
CREATE INDEX IF NOT EXISTS idx_incident_counts_daily_district_category
    ON incident_counts_daily (police_district, incident_category);

CREATE TABLE IF NOT EXISTS risk_features_hourly (
    feature_timestamp TIMESTAMP NOT NULL,
    police_district TEXT NOT NULL,
    incident_category TEXT NOT NULL,
    hour_of_day INT NOT NULL,
    day_of_week TEXT NOT NULL,
    month_of_year INT NOT NULL,
    incidents_last_1h INT NOT NULL,
    incidents_last_3h INT NOT NULL,
    incidents_last_6h INT NOT NULL,
    incidents_last_24h INT NOT NULL,
    incidents_last_7d INT NOT NULL,
    open_active_ratio_24h DOUBLE PRECISION,
    filed_online_ratio_24h DOUBLE PRECISION,
    avg_report_delay_minutes_24h DOUBLE PRECISION,
    target_risk_level TEXT,
    target_risk_score DOUBLE PRECISION,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (feature_timestamp, police_district, incident_category)
);

CREATE INDEX IF NOT EXISTS idx_risk_features_hourly_timestamp
    ON risk_features_hourly (feature_timestamp);
CREATE INDEX IF NOT EXISTS idx_risk_features_hourly_district_category
    ON risk_features_hourly (police_district, incident_category);

CREATE TABLE IF NOT EXISTS forecast_training_series (
    series_id TEXT NOT NULL,
    bucket_start TIMESTAMP NOT NULL,
    police_district TEXT NOT NULL,
    incident_category TEXT NOT NULL,
    total_incidents INT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (series_id, bucket_start)
);

CREATE INDEX IF NOT EXISTS idx_forecast_training_series_bucket_start
    ON forecast_training_series (bucket_start);
CREATE INDEX IF NOT EXISTS idx_forecast_training_series_district_category
    ON forecast_training_series (police_district, incident_category);

CREATE TABLE IF NOT EXISTS incident_hotspots (
    hotspot_id BIGSERIAL PRIMARY KEY,
    computed_at TIMESTAMP NOT NULL,
    incident_category TEXT,
    time_window_start TIMESTAMP,
    time_window_end TIMESTAMP,
    cluster_label INT,
    point_count INT,
    centroid geometry(Point, 4326),
    hotspot_geom geometry(Polygon, 4326),
    density_score DOUBLE PRECISION,
    is_hotspot BOOLEAN,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_incident_hotspots_computed_at
    ON incident_hotspots (computed_at);
CREATE INDEX IF NOT EXISTS idx_incident_hotspots_category
    ON incident_hotspots (incident_category);
CREATE INDEX IF NOT EXISTS idx_incident_hotspots_centroid
    ON incident_hotspots USING GIST (centroid);
CREATE INDEX IF NOT EXISTS idx_incident_hotspots_geom
    ON incident_hotspots USING GIST (hotspot_geom);

CREATE TABLE IF NOT EXISTS spatial_grid_features (
    grid_id TEXT NOT NULL,
    grid_geom geometry(Polygon, 4326),
    snapshot_timestamp TIMESTAMP NOT NULL,
    incidents_last_24h INT NOT NULL,
    incidents_last_7d INT NOT NULL,
    theft_ratio_7d DOUBLE PRECISION,
    assault_ratio_7d DOUBLE PRECISION,
    open_active_ratio_7d DOUBLE PRECISION,
    night_incidents_ratio_7d DOUBLE PRECISION,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (grid_id, snapshot_timestamp)
);

CREATE INDEX IF NOT EXISTS idx_spatial_grid_features_snapshot_timestamp
    ON spatial_grid_features (snapshot_timestamp);
CREATE INDEX IF NOT EXISTS idx_spatial_grid_features_geom
    ON spatial_grid_features USING GIST (grid_geom);

CREATE TABLE IF NOT EXISTS route_requests (
    route_id BIGSERIAL PRIMARY KEY,
    requested_at TIMESTAMP NOT NULL,
    origin_lat DOUBLE PRECISION,
    origin_lon DOUBLE PRECISION,
    dest_lat DOUBLE PRECISION,
    dest_lon DOUBLE PRECISION,
    route_geom geometry(LineString, 4326),
    travel_hour INT,
    travel_day_of_week TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_route_requests_requested_at
    ON route_requests (requested_at);
CREATE INDEX IF NOT EXISTS idx_route_requests_geom
    ON route_requests USING GIST (route_geom);

CREATE TABLE IF NOT EXISTS route_risk_features (
    route_feature_id BIGSERIAL PRIMARY KEY,
    route_id BIGINT REFERENCES route_requests(route_id) ON DELETE CASCADE,
    computed_at TIMESTAMP NOT NULL,
    travel_hour INT,
    travel_day_of_week TEXT,
    incidents_near_route_100m_24h INT,
    incidents_near_route_250m_24h INT,
    incidents_near_route_500m_24h INT,
    incidents_near_route_7d INT,
    theft_ratio_near_route_7d DOUBLE PRECISION,
    assault_ratio_near_route_7d DOUBLE PRECISION,
    night_ratio_near_route_7d DOUBLE PRECISION,
    avg_distance_incidents_m DOUBLE PRECISION,
    max_segment_density DOUBLE PRECISION,
    target_risk_score DOUBLE PRECISION,
    target_risk_level TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_route_risk_features_route_id
    ON route_risk_features (route_id);
CREATE INDEX IF NOT EXISTS idx_route_risk_features_computed_at
    ON route_risk_features (computed_at);

CREATE TABLE IF NOT EXISTS forecast_predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    generated_at TIMESTAMP NOT NULL,
    forecast_for TIMESTAMP NOT NULL,
    police_district TEXT,
    incident_category TEXT,
    predicted_incidents DOUBLE PRECISION,
    lower_bound DOUBLE PRECISION,
    upper_bound DOUBLE PRECISION,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_forecast_predictions_forecast_for
    ON forecast_predictions (forecast_for);
CREATE INDEX IF NOT EXISTS idx_forecast_predictions_model_name
    ON forecast_predictions (model_name);

CREATE TABLE IF NOT EXISTS risk_predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    generated_at TIMESTAMP NOT NULL,
    target_timestamp TIMESTAMP NOT NULL,
    police_district TEXT,
    incident_category TEXT,
    risk_score DOUBLE PRECISION,
    risk_level TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_risk_predictions_target_timestamp
    ON risk_predictions (target_timestamp);
CREATE INDEX IF NOT EXISTS idx_risk_predictions_model_name
    ON risk_predictions (model_name);

CREATE TABLE IF NOT EXISTS route_risk_predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    generated_at TIMESTAMP NOT NULL,
    route_id BIGINT REFERENCES route_requests(route_id) ON DELETE CASCADE,
    risk_score DOUBLE PRECISION,
    risk_level TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_route_risk_predictions_route_id
    ON route_risk_predictions (route_id);
CREATE INDEX IF NOT EXISTS idx_route_risk_predictions_model_name
    ON route_risk_predictions (model_name);

CREATE TABLE IF NOT EXISTS anomaly_detections (
    anomaly_id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    generated_at TIMESTAMP NOT NULL,
    bucket_start TIMESTAMP NOT NULL,
    police_district TEXT,
    incident_category TEXT,
    expected_min DOUBLE PRECISION,
    expected_max DOUBLE PRECISION,
    observed_value DOUBLE PRECISION,
    anomaly BOOLEAN,
    severity TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_anomaly_detections_bucket_start
    ON anomaly_detections (bucket_start);
CREATE INDEX IF NOT EXISTS idx_anomaly_detections_model_name
    ON anomaly_detections (model_name);

COMMIT;
