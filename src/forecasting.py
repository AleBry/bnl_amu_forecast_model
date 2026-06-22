from pathlib import Path
from datetime import timedelta
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_FILE = PROJECT_ROOT / "config" / "contract_config.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

HISTORICAL_WEEKLY_FILE = PROCESSED_DIR / "historical_weekly_summary.csv"
WEEKLY_SUMMARY_FILE = PROCESSED_DIR / "weekly_summary_all.csv"
CONTRACT_STATUS_FILE = PROCESSED_DIR / "contract_status_summary.csv"

OUTPUT_FILE = PROCESSED_DIR / "forecast_summary.csv"


def load_config() -> dict:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_historical_weekly_summary() -> pd.DataFrame:
    df = pd.read_csv(HISTORICAL_WEEKLY_FILE)
    df["period_start"] = pd.to_datetime(df["period_start"])
    df["period_end"] = pd.to_datetime(df["period_end"])
    return df


def load_operational_weekly_summary() -> pd.DataFrame:
    if not WEEKLY_SUMMARY_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(WEEKLY_SUMMARY_FILE)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["week_end"] = pd.to_datetime(df["week_end"])
    return df.sort_values("week_start")


def load_contract_status() -> pd.DataFrame:
    if not CONTRACT_STATUS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {CONTRACT_STATUS_FILE}. Run contract_status.py first."
        )

    df = pd.read_csv(CONTRACT_STATUS_FILE)
    df["contract_start_date"] = pd.to_datetime(df["contract_start_date"])
    df["contract_end_date"] = pd.to_datetime(df["contract_end_date"])
    df["latest_usage_date"] = pd.to_datetime(df["latest_usage_date"])
    return df


def select_auto_weights(config: dict, operational_weeks: int) -> dict:
    forecast_config = config["forecast"]
    schedule = forecast_config["auto_weight_schedule"]

    for rule in schedule:
        min_weeks = rule["min_operational_weeks"]
        max_weeks = rule.get("max_operational_weeks")

        if operational_weeks >= min_weeks and (
            max_weeks is None or operational_weeks <= max_weeks
        ):
            return {
                "historical_weight": rule.get("historical_weight"),
                "recent_average_weight": rule.get("recent_average_weight"),
                "latest_week_weight": rule.get("latest_week_weight"),
            }

    raise ValueError(
        f"No auto weight rule matched {operational_weeks} operational weeks."
    )


def normalize_active_weights(weights: dict) -> dict:
    active_weights = {
        key: value
        for key, value in weights.items()
        if value is not None and value > 0
    }

    total = sum(active_weights.values())

    if total <= 0:
        raise ValueError("No active forecast weights available.")

    return {
        key: value / total
        for key, value in active_weights.items()
    }


def calculate_forecast_components(
    historical_df: pd.DataFrame,
    operational_df: pd.DataFrame,
    config: dict,
) -> tuple[dict, dict]:
    forecast_config = config["forecast"]

    historical_avg_burn = float(historical_df["total_credits_used"].mean())

    operational_weeks = len(operational_df)

    latest_week_burn = None
    recent_average_burn = None

    if operational_weeks > 0:
        latest_week_burn = float(
            operational_df.sort_values("week_start").iloc[-1]["total_credits_used"]
        )

    recent_window = int(forecast_config.get("recent_average_window_weeks", 4))
    minimum_recent_weeks = int(
        forecast_config.get("minimum_weeks_for_recent_average", recent_window)
    )

    if operational_weeks >= minimum_recent_weeks:
        recent_average_burn = float(
            operational_df.sort_values("week_start")
            .tail(recent_window)["total_credits_used"]
            .mean()
        )

    if forecast_config.get("mode") == "auto":
        raw_weights = select_auto_weights(config, operational_weeks)
    else:
        raw_weights = {
            "historical_weight": forecast_config.get("historical_weight"),
            "recent_average_weight": forecast_config.get("recent_average_weight"),
            "latest_week_weight": forecast_config.get("latest_week_weight"),
        }

    usable_weights = {}

    if raw_weights.get("historical_weight") is not None:
        usable_weights["historical_weight"] = raw_weights["historical_weight"]

    if raw_weights.get("latest_week_weight") is not None and latest_week_burn is not None:
        usable_weights["latest_week_weight"] = raw_weights["latest_week_weight"]

    if (
        raw_weights.get("recent_average_weight") is not None
        and recent_average_burn is not None
    ):
        usable_weights["recent_average_weight"] = raw_weights["recent_average_weight"]

    if forecast_config.get("normalize_weights", True):
        final_weights = normalize_active_weights(usable_weights)
    else:
        final_weights = usable_weights

    components = {
        "historical_avg_burn": historical_avg_burn,
        "latest_week_burn": latest_week_burn,
        "recent_average_burn": recent_average_burn,
        "operational_weeks": operational_weeks,
    }

    return components, final_weights


def calculate_forecast(
    components: dict,
    weights: dict,
    contract_status: pd.DataFrame,
) -> pd.DataFrame:
    row = contract_status.iloc[0]

    forecast_weekly_burn = 0.0

    if "historical_weight" in weights:
        forecast_weekly_burn += (
            components["historical_avg_burn"] * weights["historical_weight"]
        )

    if "latest_week_weight" in weights:
        forecast_weekly_burn += (
            components["latest_week_burn"] * weights["latest_week_weight"]
        )

    if "recent_average_weight" in weights:
        forecast_weekly_burn += (
            components["recent_average_burn"] * weights["recent_average_weight"]
        )

    credits_remaining = float(row["credits_remaining"])
    latest_usage_date = pd.to_datetime(row["latest_usage_date"])
    contract_end_date = pd.to_datetime(row["contract_end_date"])
    weeks_remaining = float(row["weeks_remaining"])
    purchased_credits = float(row["purchased_credits"])
    total_credits_used = float(row["total_credits_used"])

    forecast_monthly_burn = forecast_weekly_burn * 4.345

    if forecast_weekly_burn > 0:
        weeks_until_exhaustion = credits_remaining / forecast_weekly_burn
        forecast_exhaustion_date = latest_usage_date + timedelta(
            days=weeks_until_exhaustion * 7
        )
    else:
        weeks_until_exhaustion = None
        forecast_exhaustion_date = None

    forecast_future_usage_to_contract_end = forecast_weekly_burn * weeks_remaining
    forecast_contract_end_balance = (
        credits_remaining - forecast_future_usage_to_contract_end
    )

    forecast_total_contract_usage = (
        total_credits_used + forecast_future_usage_to_contract_end
    )

    forecast_percent_credits_used_by_contract_end = (
        forecast_total_contract_usage / purchased_credits
        if purchased_credits > 0
        else 0
    )

    if forecast_contract_end_balance < 0:
        forecast_status = "EXHAUSTION_RISK"
    elif forecast_contract_end_balance <= 50000:
        forecast_status = "ON_TARGET"
    elif forecast_contract_end_balance <= 150000:
        forecast_status = "MODERATE_UNDERUSE"
    else:
        forecast_status = "HIGH_UNDERUSE"

    result = {
        "operational_weeks": components["operational_weeks"],
        "historical_avg_burn": components["historical_avg_burn"],
        "latest_week_burn": components["latest_week_burn"],
        "recent_average_burn": components["recent_average_burn"],
        "historical_weight_used": weights.get("historical_weight", 0),
        "latest_week_weight_used": weights.get("latest_week_weight", 0),
        "recent_average_weight_used": weights.get("recent_average_weight", 0),
        "forecast_weekly_burn": forecast_weekly_burn,
        "forecast_monthly_burn": forecast_monthly_burn,
        "credits_remaining": credits_remaining,
        "weeks_remaining": weeks_remaining,
        "weeks_until_exhaustion": weeks_until_exhaustion,
        "forecast_exhaustion_date": forecast_exhaustion_date.date()
        if forecast_exhaustion_date is not None
        else None,
        "contract_end_date": contract_end_date.date(),
        "forecast_future_usage_to_contract_end": forecast_future_usage_to_contract_end,
        "forecast_contract_end_balance": forecast_contract_end_balance,
        "forecast_total_contract_usage": forecast_total_contract_usage,
        "forecast_percent_credits_used_by_contract_end": forecast_percent_credits_used_by_contract_end,
        "forecast_status": forecast_status,
    }

    return pd.DataFrame([result])


def print_forecast(forecast_df: pd.DataFrame) -> None:
    row = forecast_df.iloc[0]

    print("\nForecast Summary")
    print("----------------")
    print(f"Operational weeks available:       {row['operational_weeks']}")
    print(f"Historical average weekly burn:    {row['historical_avg_burn']:,.2f}")
    print(f"Latest weekly burn:                {row['latest_week_burn']:,.2f}")

    if pd.notna(row["recent_average_burn"]):
        print(f"Recent average burn:               {row['recent_average_burn']:,.2f}")
    else:
        print("Recent average burn:               Not enough weekly uploads yet")

    print("\nWeights Used")
    print("------------")
    print(f"Historical weight:                 {row['historical_weight_used']:.2f}")
    print(f"Latest week weight:                {row['latest_week_weight_used']:.2f}")
    print(f"Recent average weight:             {row['recent_average_weight_used']:.2f}")

    print("\nProjection")
    print("----------")
    print(f"Forecast weekly burn:              {row['forecast_weekly_burn']:,.2f}")
    print(f"Forecast monthly burn:             {row['forecast_monthly_burn']:,.2f}")
    print(f"Credits remaining:                 {row['credits_remaining']:,.2f}")
    print(f"Weeks remaining:                   {row['weeks_remaining']:,.2f}")
    print(f"Weeks until exhaustion:            {row['weeks_until_exhaustion']:,.2f}")
    print(f"Forecast exhaustion date:          {row['forecast_exhaustion_date']}")
    print(f"Contract end date:                 {row['contract_end_date']}")
    print(f"Forecast contract-end balance:     {row['forecast_contract_end_balance']:,.2f}")
    print(f"Forecast percent credits used:     {row['forecast_percent_credits_used_by_contract_end']:.1%}")
    print(f"Forecast status:                   {row['forecast_status']}")


def main() -> None:
    config = load_config()
    historical_df = load_historical_weekly_summary()
    operational_df = load_operational_weekly_summary()
    contract_status = load_contract_status()

    components, weights = calculate_forecast_components(
        historical_df=historical_df,
        operational_df=operational_df,
        config=config,
    )

    forecast_df = calculate_forecast(
        components=components,
        weights=weights,
        contract_status=contract_status,
    )

    forecast_df.to_csv(OUTPUT_FILE, index=False)

    print_forecast(forecast_df)

    print("\nFile written")
    print("------------")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()