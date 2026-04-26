"""
Weekly ML training pipeline runner for CI San Francisco.

This pipeline currently runs only the volume model retraining script.
Both files can live in the same directory:
- run_pipeline.py
- ml_model_volume.py

Usage:
    python run_pipeline.py

Optional env vars:
    ML_LOOKBACK_DAYS=180
    ML_TEST_SIZE=0.2
    WEEKLY_TRAINING_TIMEOUT_SECONDS=3600
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
TRAINING_SCRIPT = CURRENT_DIR / "ml_model_volume.py"
DEFAULT_TIMEOUT_SECONDS = 3600


def parse_timeout() -> int:
    raw = os.environ.get("WEEKLY_TRAINING_TIMEOUT_SECONDS")
    try:
        value = int(raw) if raw is not None else DEFAULT_TIMEOUT_SECONDS
    except ValueError:
        value = DEFAULT_TIMEOUT_SECONDS
    return max(60, min(value, 7200))


def run_step(name: str, command: list[str], timeout_seconds: int) -> dict[str, Any]:
    started_at = datetime.utcnow().replace(microsecond=0)
    print(
        json.dumps(
            {
                "status": "running",
                "step": name,
                "started_at": started_at.isoformat(),
                "command": command,
            }
        ),
        flush=True,
    )

    completed = subprocess.run(
        command,
        cwd=str(CURRENT_DIR),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )

    finished_at = datetime.utcnow().replace(microsecond=0)
    result = {
        "step": name,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "returncode": completed.returncode,
        "stdout": completed.stdout[-20000:],
        "stderr": completed.stderr[-20000:],
    }

    if completed.returncode != 0:
        raise RuntimeError(json.dumps(result, indent=2, default=str))

    print(json.dumps({"status": "ok", **result}, indent=2, default=str), flush=True)
    return result


def main() -> dict[str, Any]:
    if not TRAINING_SCRIPT.exists():
        raise FileNotFoundError(f"Training script not found: {TRAINING_SCRIPT}")

    timeout_seconds = parse_timeout()
    started_at = datetime.utcnow().replace(microsecond=0)

    print(
        json.dumps(
            {
                "status": "running",
                "pipeline": "weekly_model_training",
                "started_at": started_at.isoformat(),
                "timeout_seconds": timeout_seconds,
                "included_models": ["volume_random_forest_v1"],
            }
        ),
        flush=True,
    )

    steps = [
        run_step(
            name="ml_model_volume",
            command=[sys.executable, str(TRAINING_SCRIPT)],
            timeout_seconds=timeout_seconds,
        )
    ]

    finished_at = datetime.utcnow().replace(microsecond=0)
    result = {
        "status": "ok",
        "pipeline": "weekly_model_training",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "steps": steps,
    }
    print(json.dumps(result, indent=2, default=str), flush=True)
    return result


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error = {
            "status": "error",
            "pipeline": "weekly_model_training",
            "message": str(exc),
        }
        print(json.dumps(error, indent=2, default=str), file=sys.stderr, flush=True)
        raise
