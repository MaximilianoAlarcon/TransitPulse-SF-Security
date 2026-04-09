import subprocess
import sys
from datetime import datetime

SCRIPTS = [
    "scripts/load_incidents.py",
    "scripts/refresh_hourly_aggregates.py",
    "scripts/refresh_daily_aggregates.py",
    "scripts/build_forecast_series.py",
    "scripts/build_risk_features.py",
]

def run_script(script):
    print(f"\n🚀 Running: {script}")
    start = datetime.now()

    result = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True
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

def main():
    print("===== START PIPELINE =====")

    for script in SCRIPTS:
        run_script(script)

    print("===== PIPELINE COMPLETED =====")

if __name__ == "__main__":
    main()