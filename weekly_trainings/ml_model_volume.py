"""
Weekly training entrypoint for the CI San Francisco volume model.

This script reuses the production ML functions already defined in app.py:
- fetch_volume_ml_dataset
- summarize_volume_ml_dataset
- train_volume_forecast_model<

It trains the volume_random_forest_v1 model and persists artifacts using the same
storage logic as the Flask admin endpoint, including Railway Bucket / S3 upload
when AWS_* variables are configured.

Usage:
    python retrain_volume_model.py

Optional env vars:
    ML_LOOKBACK_DAYS=180
    ML_TEST_SIZE=0.2
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR

# Support both layouts:
# 1) project_root/retrain_volume_model.py
# 2) project_root/training_scripts/retrain_volume_model.py
if not (PROJECT_ROOT / "app.py").exists() and (PROJECT_ROOT.parent / "app.py").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_int_env(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.environ.get(name)
    try:
        value = int(raw) if raw is not None else default
    except ValueError:
        value = default
    return max(min_value, min(value, max_value))


def parse_float_env(name: str, default: float, min_value: float, max_value: float) -> float:
    raw = os.environ.get(name)
    try:
        value = float(raw) if raw is not None else default
    except ValueError:
        value = default
    return max(min_value, min(value, max_value))


def main() -> dict[str, Any]:
    try:
        from app import (  # type: ignore
            DEFAULT_ML_LOOKBACK_DAYS,
            DEFAULT_ML_TEST_SIZE,
            fetch_volume_ml_dataset,
            summarize_volume_ml_dataset,
            train_volume_forecast_model,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not import ML training functions from app.py. "
            "Place this script in the project root or in a child folder such as training_scripts/."
        ) from exc

    lookback_days = parse_int_env(
        "ML_LOOKBACK_DAYS",
        default=int(DEFAULT_ML_LOOKBACK_DAYS),
        min_value=7,
        max_value=3650,
    )
    test_size = parse_float_env(
        "ML_TEST_SIZE",
        default=float(DEFAULT_ML_TEST_SIZE),
        min_value=0.05,
        max_value=0.4,
    )

    started_at = datetime.utcnow().replace(microsecond=0)
    print(
        json.dumps(
            {
                "status": "running",
                "step": "fetch_dataset",
                "started_at": started_at.isoformat(),
                "lookback_days": lookback_days,
                "test_size": test_size,
            }
        ),
        flush=True,
    )

    rows = fetch_volume_ml_dataset(lookback_days=lookback_days)
    dataset_summary = summarize_volume_ml_dataset(rows, lookback_days=lookback_days)

    print(
        json.dumps(
            {
                "status": "running",
                "step": "train_model",
                "dataset": dataset_summary,
            },
            default=str,
        ),
        flush=True,
    )

    training_result = train_volume_forecast_model(rows, test_size=test_size)
    finished_at = datetime.utcnow().replace(microsecond=0)

    result = {
        "status": "ok",
        "pipeline": "weekly_volume_model_training",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "dataset": dataset_summary,
        "training": training_result,
    }

    print(json.dumps(result, indent=2, default=str), flush=True)
    return result


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error = {
            "status": "error",
            "pipeline": "weekly_volume_model_training",
            "message": str(exc),
        }
        print(json.dumps(error, indent=2, default=str), file=sys.stderr, flush=True)
        raise
