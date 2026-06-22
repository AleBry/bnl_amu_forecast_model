from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_FILE = PROJECT_ROOT / "config" / "contract_config.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CONTRACT_STATUS_FILE = PROCESSED_DIR / "contract_status_summary.csv"
FORECAST_FILE = PROCESSED_DIR / "forecast_summary.csv"
OPERATIONAL_WEEKLY_FILE = PROCESSED_DIR / "weekly_summary_all.csv"
HISTORICAL_WEEKLY_FILE = PROCESSED_DIR / "historical_weekly_summary.csv"

SUMMARY_OUTPUT = PROCESSED_DIR / "monte_carlo_summary.csv"
DISTRIBUTION_OUTPUT = PROCESSED_DIR / "monte_carlo_distribution.csv"
WEEKLY_SIMULATION_OUTPUT = PROCESSED_DIR / "monte_carlo_weekly_simulation.csv"

# Model implementation detail, intentionally kept in code.
MIN_OBSERVATIONS_FOR_EMPIRICAL_SAMPLING = 4


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_FILE}")

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_contract_status() -> pd.DataFrame:
    if not CONTRACT_STATUS_FILE.exists():
        raise FileNotFoundError(
            f"Missing contract status file: {CONTRACT_STATUS_FILE}. "
            "Run contract_status.py first."
        )

    df = pd.read_csv(CONTRACT_STATUS_FILE)
    df["contract_start_date"] = pd.to_datetime(df["contract_start_date"])
    df["contract_end_date"] = pd.to_datetime(df["contract_end_date"])
    df["latest_usage_date"] = pd.to_datetime(df["latest_usage_date"])
    return df


def load_forecast() -> pd.DataFrame:
    if not FORECAST_FILE.exists():
        raise FileNotFoundError(
            f"Missing forecast file: {FORECAST_FILE}. Run forecasting.py first."
        )

    df = pd.read_csv(FORECAST_FILE)
    df["contract_end_date"] = pd.to_datetime(df["contract_end_date"])
    return df


def load_weekly_burn_file(file_path: Path, date_column: str | None = None) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(file_path)

    if "total_credits_used" not in df.columns:
        raise ValueError(
            f"Expected column 'total_credits_used' in {file_path}, "
            f"but found columns: {list(df.columns)}"
        )

    df["total_credits_used"] = pd.to_numeric(
        df["total_credits_used"],
        errors="coerce",
    ).fillna(0)

    if date_column and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)

    return df


def require_config_value(config_section: dict, key: str, section_name: str):
    if key not in config_section or config_section[key] is None:
        raise ValueError(
            f"Missing required config value: {section_name}.{key}"
        )
    return config_section[key]


def get_monte_carlo_config(config: dict) -> dict:
    if "monte_carlo" not in config or config["monte_carlo"] is None:
        raise ValueError(
            "Missing required 'monte_carlo' section in contract_config.yaml. "
            "Expected keys: runs, random_seed, stranding_threshold_credits."
        )

    monte_carlo_config = config["monte_carlo"]

    runs = int(require_config_value(monte_carlo_config, "runs", "monte_carlo"))
    stranding_threshold_credits = float(
        require_config_value(
            monte_carlo_config,
            "stranding_threshold_credits",
            "monte_carlo",
        )
    )

    random_seed = monte_carlo_config.get("random_seed")
    if random_seed is not None:
        random_seed = int(random_seed)

    if runs <= 0:
        raise ValueError("monte_carlo.runs must be greater than zero.")

    if stranding_threshold_credits < 0:
        raise ValueError("monte_carlo.stranding_threshold_credits must be non-negative.")

    return {
        "runs": runs,
        "stranding_threshold_credits": stranding_threshold_credits,
        "random_seed": random_seed,
    }


def build_burn_observations() -> tuple[pd.Series, str]:
    operational_df = load_weekly_burn_file(OPERATIONAL_WEEKLY_FILE, "week_start")

    if len(operational_df) >= MIN_OBSERVATIONS_FOR_EMPIRICAL_SAMPLING:
        observations = operational_df["total_credits_used"]
        return observations[observations >= 0], "operational_weekly_summary"

    historical_df = load_weekly_burn_file(HISTORICAL_WEEKLY_FILE, "period_start")

    if not historical_df.empty:
        observations = historical_df["total_credits_used"]

        if not operational_df.empty:
            observations = pd.concat(
                [observations, operational_df["total_credits_used"]],
                ignore_index=True,
            )
            return observations[observations >= 0], "historical_plus_operational_weekly_summary"

        return observations[observations >= 0], "historical_weekly_summary"

    if not operational_df.empty:
        observations = operational_df["total_credits_used"]
        return observations[observations >= 0], "operational_weekly_summary_limited"

    return pd.Series(dtype="float64"), "fallback_constant_forecast"


def build_empirical_multipliers(
    observations: pd.Series,
    forecast_weekly_burn: float,
) -> np.ndarray:
    clean = pd.to_numeric(observations, errors="coerce").dropna()
    clean = clean[clean >= 0]

    if len(clean) < 2 or clean.mean() <= 0:
        return np.array([1.0])

    multipliers = clean / clean.mean()
    multipliers = multipliers.replace([np.inf, -np.inf], np.nan).dropna()

    if len(multipliers) == 0:
        return np.array([1.0])

    # Keep the distribution centered on the deterministic forecast.
    # This uses observed volatility but preserves forecasting.py as the expected burn level.
    return multipliers.to_numpy(dtype=float)


def simulate_one_run(
    run_id: int,
    rng: np.random.Generator,
    multipliers: np.ndarray,
    forecast_weekly_burn: float,
    credits_remaining: float,
    weeks_remaining: float,
    latest_usage_date: pd.Timestamp,
) -> tuple[dict, list[dict]]:
    full_weeks = int(np.floor(weeks_remaining))
    partial_week_fraction = float(weeks_remaining - full_weeks)

    week_fractions = [1.0] * full_weeks

    if partial_week_fraction > 0:
        week_fractions.append(partial_week_fraction)

    cumulative_usage = 0.0
    exhausted = False
    exhaustion_date = None
    exhaustion_days_from_latest_usage = None
    weekly_rows = []

    for week_index, week_fraction in enumerate(week_fractions, start=1):
        multiplier = float(rng.choice(multipliers))
        simulated_weekly_burn = max(forecast_weekly_burn * multiplier, 0)
        simulated_period_burn = simulated_weekly_burn * week_fraction

        previous_cumulative_usage = cumulative_usage
        cumulative_usage += simulated_period_burn

        week_start_date = latest_usage_date + timedelta(days=(week_index - 1) * 7)
        week_end_date = week_start_date + timedelta(days=7 * week_fraction)

        if not exhausted and cumulative_usage >= credits_remaining:
            exhausted = True

            burn_needed_this_period = credits_remaining - previous_cumulative_usage

            if simulated_period_burn > 0:
                fraction_into_period = burn_needed_this_period / simulated_period_burn
                fraction_into_period = min(max(fraction_into_period, 0), 1)
            else:
                fraction_into_period = 1

            days_into_period = 7 * week_fraction * fraction_into_period
            exhaustion_days_from_latest_usage = ((week_index - 1) * 7) + days_into_period
            exhaustion_date = latest_usage_date + timedelta(
                days=exhaustion_days_from_latest_usage
            )

        weekly_rows.append(
            {
                "run_id": run_id,
                "simulation_week": week_index,
                "week_start_date": week_start_date.date(),
                "week_end_date": week_end_date.date(),
                "week_fraction": week_fraction,
                "burn_multiplier": multiplier,
                "simulated_weekly_burn": simulated_weekly_burn,
                "simulated_period_burn": simulated_period_burn,
                "cumulative_future_usage": cumulative_usage,
            }
        )

    contract_end_balance = credits_remaining - cumulative_usage

    run_result = {
        "run_id": run_id,
        "future_weeks_simulated": weeks_remaining,
        "simulated_future_usage": cumulative_usage,
        "contract_end_balance": contract_end_balance,
        "exhausted": exhausted,
        "exhaustion_days_from_latest_usage": exhaustion_days_from_latest_usage,
        "exhaustion_date": exhaustion_date.date() if exhaustion_date is not None else None,
    }

    return run_result, weekly_rows


def run_monte_carlo(
    config: dict,
    contract_status: pd.DataFrame,
    forecast: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mc_config = get_monte_carlo_config(config)

    status_row = contract_status.iloc[0]
    forecast_row = forecast.iloc[0]

    credits_remaining = float(status_row["credits_remaining"])
    weeks_remaining = float(status_row["weeks_remaining"])
    latest_usage_date = pd.to_datetime(status_row["latest_usage_date"])
    contract_end_date = pd.to_datetime(status_row["contract_end_date"])
    purchased_credits = float(status_row["purchased_credits"])
    total_credits_used = float(status_row["total_credits_used"])

    forecast_weekly_burn = float(forecast_row["forecast_weekly_burn"])

    observations, burn_distribution_source = build_burn_observations()
    multipliers = build_empirical_multipliers(
        observations=observations,
        forecast_weekly_burn=forecast_weekly_burn,
    )

    rng = np.random.default_rng(mc_config["random_seed"])

    run_rows = []
    weekly_rows = []

    for run_id in range(1, mc_config["runs"] + 1):
        run_result, run_weekly_rows = simulate_one_run(
            run_id=run_id,
            rng=rng,
            multipliers=multipliers,
            forecast_weekly_burn=forecast_weekly_burn,
            credits_remaining=credits_remaining,
            weeks_remaining=weeks_remaining,
            latest_usage_date=latest_usage_date,
        )

        run_rows.append(run_result)
        weekly_rows.extend(run_weekly_rows)

    distribution_df = pd.DataFrame(run_rows)
    weekly_simulation_df = pd.DataFrame(weekly_rows)

    stranding_threshold = mc_config["stranding_threshold_credits"]

    distribution_df["stranded"] = (
        distribution_df["contract_end_balance"] > stranding_threshold
    )

    distribution_df["forecast_total_contract_usage"] = (
        total_credits_used + distribution_df["simulated_future_usage"]
    )

    distribution_df["forecast_percent_credits_used_by_contract_end"] = (
        distribution_df["forecast_total_contract_usage"] / purchased_credits
        if purchased_credits > 0
        else 0
    )

    summary_df = build_summary(
        distribution_df=distribution_df,
        observations=observations,
        multipliers=multipliers,
        mc_config=mc_config,
        burn_distribution_source=burn_distribution_source,
        forecast_weekly_burn=forecast_weekly_burn,
        credits_remaining=credits_remaining,
        weeks_remaining=weeks_remaining,
        latest_usage_date=latest_usage_date,
        contract_end_date=contract_end_date,
        purchased_credits=purchased_credits,
        total_credits_used=total_credits_used,
    )

    return summary_df, distribution_df, weekly_simulation_df


def percentile(series: pd.Series, q: float) -> float:
    return float(series.quantile(q / 100))


def percentile_exhaustion_date(
    days_from_latest_usage: pd.Series,
    q: float,
    latest_usage_date: pd.Timestamp,
):
    clean = pd.to_numeric(days_from_latest_usage, errors="coerce").dropna()

    if clean.empty:
        return None

    percentile_days = float(np.percentile(clean, q))
    return (latest_usage_date + timedelta(days=percentile_days)).date()


def build_summary(
    distribution_df: pd.DataFrame,
    observations: pd.Series,
    multipliers: np.ndarray,
    mc_config: dict,
    burn_distribution_source: str,
    forecast_weekly_burn: float,
    credits_remaining: float,
    weeks_remaining: float,
    latest_usage_date: pd.Timestamp,
    contract_end_date: pd.Timestamp,
    purchased_credits: float,
    total_credits_used: float,
) -> pd.DataFrame:
    exhaustion_days = distribution_df.loc[
        distribution_df["exhausted"],
        "exhaustion_days_from_latest_usage",
    ]

    summary = {
        "monte_carlo_runs": mc_config["runs"],
        "random_seed": mc_config["random_seed"],
        "burn_distribution_source": burn_distribution_source,
        "burn_observations_used": len(observations),
        "burn_observation_mean": float(observations.mean()) if len(observations) > 0 else None,
        "burn_observation_std": float(observations.std()) if len(observations) > 1 else None,
        "burn_multiplier_mean": float(np.mean(multipliers)),
        "burn_multiplier_std": float(np.std(multipliers)),
        "latest_usage_date": latest_usage_date.date(),
        "contract_end_date": contract_end_date.date(),
        "purchased_credits": purchased_credits,
        "total_credits_used_to_date": total_credits_used,
        "credits_remaining_at_simulation_start": credits_remaining,
        "weeks_remaining_at_simulation_start": weeks_remaining,
        "forecast_weekly_burn_center": forecast_weekly_burn,
        "stranding_threshold_credits": mc_config["stranding_threshold_credits"],
        "exhaustion_probability": float(distribution_df["exhausted"].mean()),
        "stranding_probability": float(distribution_df["stranded"].mean()),
        "p10_future_usage": percentile(distribution_df["simulated_future_usage"], 10),
        "p50_future_usage": percentile(distribution_df["simulated_future_usage"], 50),
        "p90_future_usage": percentile(distribution_df["simulated_future_usage"], 90),
        "p10_contract_end_balance": percentile(distribution_df["contract_end_balance"], 10),
        "p50_contract_end_balance": percentile(distribution_df["contract_end_balance"], 50),
        "p90_contract_end_balance": percentile(distribution_df["contract_end_balance"], 90),
        "p10_percent_credits_used_by_contract_end": percentile(
            distribution_df["forecast_percent_credits_used_by_contract_end"],
            10,
        ),
        "p50_percent_credits_used_by_contract_end": percentile(
            distribution_df["forecast_percent_credits_used_by_contract_end"],
            50,
        ),
        "p90_percent_credits_used_by_contract_end": percentile(
            distribution_df["forecast_percent_credits_used_by_contract_end"],
            90,
        ),
        "p10_exhaustion_date": percentile_exhaustion_date(
            exhaustion_days,
            10,
            latest_usage_date,
        ),
        "p50_exhaustion_date": percentile_exhaustion_date(
            exhaustion_days,
            50,
            latest_usage_date,
        ),
        "p90_exhaustion_date": percentile_exhaustion_date(
            exhaustion_days,
            90,
            latest_usage_date,
        ),
    }

    if summary["exhaustion_probability"] >= 0.50:
        risk_status = "HIGH_EXHAUSTION_RISK"
    elif summary["exhaustion_probability"] >= 0.20:
        risk_status = "MODERATE_EXHAUSTION_RISK"
    elif summary["stranding_probability"] >= 0.50:
        risk_status = "HIGH_STRANDING_RISK"
    elif summary["stranding_probability"] >= 0.20:
        risk_status = "MODERATE_STRANDING_RISK"
    else:
        risk_status = "BALANCED_RISK"

    summary["monte_carlo_risk_status"] = risk_status

    return pd.DataFrame([summary])


def save_outputs(
    summary_df: pd.DataFrame,
    distribution_df: pd.DataFrame,
    weekly_simulation_df: pd.DataFrame,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(SUMMARY_OUTPUT, index=False)
    distribution_df.to_csv(DISTRIBUTION_OUTPUT, index=False)
    weekly_simulation_df.to_csv(WEEKLY_SIMULATION_OUTPUT, index=False)


def print_summary(summary_df: pd.DataFrame) -> None:
    row = summary_df.iloc[0]

    print("\nMonte Carlo Simulation Complete")
    print("-------------------------------")
    print(f"Runs:                              {row['monte_carlo_runs']:,.0f}")
    print(f"Random seed:                       {row['random_seed']}")
    print(f"Burn distribution source:          {row['burn_distribution_source']}")
    print(f"Burn observations used:            {row['burn_observations_used']:,.0f}")
    print(f"Forecast weekly burn center:       {row['forecast_weekly_burn_center']:,.2f}")
    print(f"Credits remaining at start:        {row['credits_remaining_at_simulation_start']:,.2f}")
    print(f"Weeks remaining at start:          {row['weeks_remaining_at_simulation_start']:,.2f}")

    print("\nRisk")
    print("----")
    print(f"Exhaustion probability:            {row['exhaustion_probability']:.1%}")
    print(f"Stranding probability:             {row['stranding_probability']:.1%}")
    print(f"Risk status:                       {row['monte_carlo_risk_status']}")

    print("\nContract-End Balance")
    print("--------------------")
    print(f"P10 balance:                       {row['p10_contract_end_balance']:,.2f}")
    print(f"P50 balance:                       {row['p50_contract_end_balance']:,.2f}")
    print(f"P90 balance:                       {row['p90_contract_end_balance']:,.2f}")

    print("\nExhaustion Date, Conditional on Exhaustion")
    print("------------------------------------------")
    print(f"P10 exhaustion date:               {row['p10_exhaustion_date']}")
    print(f"P50 exhaustion date:               {row['p50_exhaustion_date']}")
    print(f"P90 exhaustion date:               {row['p90_exhaustion_date']}")

    print("\nFiles written")
    print("-------------")
    print(SUMMARY_OUTPUT)
    print(DISTRIBUTION_OUTPUT)
    print(WEEKLY_SIMULATION_OUTPUT)


def main() -> None:
    config = load_config()
    contract_status = load_contract_status()
    forecast = load_forecast()

    summary_df, distribution_df, weekly_simulation_df = run_monte_carlo(
        config=config,
        contract_status=contract_status,
        forecast=forecast,
    )

    save_outputs(
        summary_df=summary_df,
        distribution_df=distribution_df,
        weekly_simulation_df=weekly_simulation_df,
    )

    print_summary(summary_df)


if __name__ == "__main__":
    main()
