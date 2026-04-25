import json
import os
import subprocess
import sys
from datetime import datetime


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


SCRIPTS = [
    "scripts/load_incidents.py",
    "scripts/refresh_hourly_aggregates.py",
    "scripts/refresh_daily_aggregates.py",
    "scripts/build_forecast_series.py",
    "scripts/build_risk_features.py",
    "scripts/build_predictions.py"
]


def build_script_env(mode: str) -> dict[str, str]:
    env = os.environ.copy()

    # filtro de categorias
    env["CATEGORY_FILTER_VALUES_JSON"] = json.dumps(CATEGORY_FILTER_VALUES)

    if mode == "backfill":
        print("⚠️ Running BACKFILL mode (6 months)")

        env["FORCE_HISTORY_BACKFILL"] = "true"
        env["HISTORY_LOOKBACK_MONTHS"] = "6"
        env["HISTORY_LOOKBACK_INTERVAL"] = "6 months"

    else:
        print("⚡ Running INCREMENTAL mode (48 hours)")

        env["FORCE_HISTORY_BACKFILL"] = "false"
        env["HISTORY_LOOKBACK_INTERVAL"] = "48 hours"

    return env


def run_script(script: str, env: dict[str, str]) -> None:
    print(f"\n🚀 Running: {script}")
    start = datetime.now()

    result = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        env=env,
    )

    duration = (datetime.now() - start).total_seconds()

    print(f"⏱️ Duration: {duration:.2f}s")

    if result.stdout:
        print("📤 STDOUT:")
        print(result.stdout)

    if result.stderr:
        print("⚠️ STDERR:")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"❌ Script failed: {script}")


def main() -> None:
    mode = os.environ.get("PIPELINE_MODE", "incremental")

    env = build_script_env(mode)

    print("===== START PIPELINE =====")
    print(f"Mode: {mode}")

    for script in SCRIPTS:
        run_script(script, env)

    print("===== PIPELINE COMPLETED =====")


if __name__ == "__main__":
    main()