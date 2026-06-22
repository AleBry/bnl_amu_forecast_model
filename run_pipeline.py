from pathlib import Path
import argparse
import subprocess
import sys
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
WEEKLY_UPLOADS_DIR = PROJECT_ROOT / "data" / "weekly_uploads"

PIPELINE_LOG = PROCESSED_DIR / "pipeline_run_log.txt"


# Historical ingestion is usually fixed/authoritative and does not need to run
# every week unless the historical workbook changes or the historical outputs
# are missing.
HISTORICAL_SCRIPT = "ingest_historical.py"

WEEKLY_PIPELINE_SCRIPTS = [
    "ingest_weekly.py",
    "contract_status.py",
    "forecasting.py",
    "user_segmentation.py",
    "cap_pressure.py",
    "cap_pressure_history.py",
    "tier_recommendations.py",
    "monte_carlo.py",
    "policy_scenario_sandbox.py",
]


REQUIRED_HISTORICAL_OUTPUTS = [
    PROCESSED_DIR / "historical_usage_cleaned.csv",
    PROCESSED_DIR / "historical_weekly_summary.csv",
    PROCESSED_DIR / "historical_user_summary.csv",
]


def historical_outputs_exist() -> bool:
    return all(path.exists() for path in REQUIRED_HISTORICAL_OUTPUTS)


def validate_project_structure() -> None:
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Missing source directory: {SRC_DIR}")

    if not WEEKLY_UPLOADS_DIR.exists():
        raise FileNotFoundError(
            f"Missing weekly uploads directory: {WEEKLY_UPLOADS_DIR}"
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def validate_scripts_exist(script_names: list[str]) -> None:
    missing = [
        str(SRC_DIR / script_name)
        for script_name in script_names
        if not (SRC_DIR / script_name).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required pipeline script(s):\n"
            + "\n".join(missing)
        )


def append_log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"

    print(line)

    with open(PIPELINE_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_script(script_name: str) -> None:
    script_path = SRC_DIR / script_name

    append_log(f"START {script_name}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        append_log(f"STDOUT {script_name}\n{result.stdout.strip()}")

    if result.stderr:
        append_log(f"STDERR {script_name}\n{result.stderr.strip()}")

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} failed with exit code {result.returncode}. "
            f"See pipeline log: {PIPELINE_LOG}"
        )

    append_log(f"COMPLETE {script_name}")


def build_pipeline_scripts(include_historical: bool) -> list[str]:
    scripts = []

    if include_historical or not historical_outputs_exist():
        scripts.append(HISTORICAL_SCRIPT)

    scripts.extend(WEEKLY_PIPELINE_SCRIPTS)

    return scripts


def run_pipeline(include_historical: bool) -> None:
    validate_project_structure()

    scripts = build_pipeline_scripts(include_historical=include_historical)

    validate_scripts_exist(scripts)

    append_log("========================================")
    append_log("BNL AMU forecasting pipeline started")
    append_log(f"Project root: {PROJECT_ROOT}")
    append_log(f"Weekly uploads directory: {WEEKLY_UPLOADS_DIR}")
    append_log(f"Processed output directory: {PROCESSED_DIR}")
    append_log(f"Include historical ingestion: {include_historical}")
    append_log("Scripts to run: " + ", ".join(scripts))

    for script_name in scripts:
        run_script(script_name)

    append_log("BNL AMU forecasting pipeline completed successfully")
    append_log("========================================")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full BNL AMU credit forecasting, governance, risk, "
            "and policy scenario pipeline."
        )
    )

    parser.add_argument(
        "--include-historical",
        action="store_true",
        help=(
            "Force rerun of ingest_historical.py. By default, historical "
            "ingestion only runs if required historical processed outputs "
            "are missing."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_pipeline(
        include_historical=args.include_historical,
    )


if __name__ == "__main__":
    main()
