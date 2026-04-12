import json
import os
import subprocess
import sys
from datetime import datetime

# Centralized category allowlist for crime-focused ETLs.
# Edit this list here and rerun the pipeline.
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
]


def build_script_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CATEGORY_FILTER_VALUES_JSON"] = json.dumps(CATEGORY_FILTER_VALUES)
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
    env = build_script_env()

    print("===== START PIPELINE =====")
    print("Crime category filter enabled with these categories:")
    for category in CATEGORY_FILTER_VALUES:
        print(f" - {category}")

    for script in SCRIPTS:
        run_script(script, env)

    print("===== PIPELINE COMPLETED =====")


if __name__ == "__main__":
    main()
