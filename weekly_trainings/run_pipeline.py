"""
Weekly ML training pipeline runner for CI San Francisco.

This runner executes a configurable list of training scripts in sequence.
All files can live in the same directory:
- run_pipeline.py
- ml_model_volume.py
- ml_model_risk_classifier.py
- ml_risk_route.py

Usage:
    python run_pipeline.py

Optional env vars:
    WEEKLY_TRAINING_TIMEOUT_SECONDS=7200
    WEEKLY_TRAINING_SKIP_DISABLED=true

To add future models, append a new dict to TRAINING_STEPS.
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
DEFAULT_TIMEOUT_SECONDS = 7200
MAX_TIMEOUT_SECONDS = 10800

# Add model-training scripts here.
# Each step runs sequentially. If one fails, the pipeline stops.
TRAINING_STEPS: list[dict[str, Any]] = [
    {
        "name": "ml_risk_route",
        "script": "ml_risk_route.py",
        "enabled": True,
        "model_name": "ml_risk_route",
        # Optional per-step timeout. If None, uses WEEKLY_TRAINING_TIMEOUT_SECONDS.
        "timeout_seconds": None,
    }
]


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


def parse_timeout(raw_value: Any, default: int = DEFAULT_TIMEOUT_SECONDS) -> int:
    try:
        value = int(raw_value) if raw_value is not None else default
    except (TypeError, ValueError):
        value = default
    return max(60, min(value, MAX_TIMEOUT_SECONDS))


def get_default_timeout_seconds() -> int:
    return parse_timeout(os.environ.get("WEEKLY_TRAINING_TIMEOUT_SECONDS"), DEFAULT_TIMEOUT_SECONDS)


def resolve_step_script(script_name: str) -> Path:
    script_path = CURRENT_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")
    if not script_path.is_file():
        raise FileNotFoundError(f"Training script path is not a file: {script_path}")
    return script_path


def enabled_steps() -> list[dict[str, Any]]:
    return [step for step in TRAINING_STEPS if bool(step.get("enabled", True))]


def run_step(step: dict[str, Any], default_timeout_seconds: int) -> dict[str, Any]:
    name = str(step["name"])
    script_path = resolve_step_script(str(step["script"]))
    timeout_seconds = parse_timeout(step.get("timeout_seconds"), default_timeout_seconds)
    command = [sys.executable, str(script_path)]

    started_at = utc_now_iso()
    print(
        json.dumps(
            {
                "status": "running",
                "step": name,
                "model_name": step.get("model_name"),
                "started_at": started_at,
                "timeout_seconds": timeout_seconds,
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

    finished_at = utc_now_iso()
    result = {
        "step": name,
        "model_name": step.get("model_name"),
        "script": str(script_path.name),
        "started_at": started_at,
        "finished_at": finished_at,
        "returncode": completed.returncode,
        "stdout": (completed.stdout or "")[-30000:],
        "stderr": (completed.stderr or "")[-30000:],
    }

    # Compute duration safely from datetime parsing.
    started_dt = datetime.fromisoformat(started_at)
    finished_dt = datetime.fromisoformat(finished_at)
    result["duration_seconds"] = round((finished_dt - started_dt).total_seconds(), 3)

    if completed.returncode != 0:
        result["status"] = "error"
        raise RuntimeError(json.dumps(result, indent=2, default=str))

    result["status"] = "ok"
    print(json.dumps(result, indent=2, default=str), flush=True)
    return result


def main() -> dict[str, Any]:
    default_timeout_seconds = get_default_timeout_seconds()
    steps_to_run = enabled_steps()
    started_at = utc_now_iso()

    if not steps_to_run:
        raise RuntimeError("No enabled training steps configured in TRAINING_STEPS.")

    print(
        json.dumps(
            {
                "status": "running",
                "pipeline": "weekly_model_training",
                "started_at": started_at,
                "default_timeout_seconds": default_timeout_seconds,
                "step_count": len(steps_to_run),
                "included_models": [step.get("model_name") for step in steps_to_run],
                "included_steps": [step.get("name") for step in steps_to_run],
            }
        ),
        flush=True,
    )

    completed_steps: list[dict[str, Any]] = []
    for step in steps_to_run:
        completed_steps.append(run_step(step, default_timeout_seconds=default_timeout_seconds))

    finished_at = utc_now_iso()
    started_dt = datetime.fromisoformat(started_at)
    finished_dt = datetime.fromisoformat(finished_at)

    result = {
        "status": "ok",
        "pipeline": "weekly_model_training",
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round((finished_dt - started_dt).total_seconds(), 3),
        "step_count": len(completed_steps),
        "steps": completed_steps,
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
