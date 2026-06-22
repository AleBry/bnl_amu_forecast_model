from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_FILE = PROJECT_ROOT / "config" / "contract_config.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

HISTORICAL_WEEKLY_FILE = PROCESSED_DIR / "historical_weekly_summary.csv"
WEEKLY_SUMMARY_FILE = PROCESSED_DIR / "weekly_summary_all.csv"

OUTPUT_FILE = PROCESSED_DIR / "contract_status_summary.csv"


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_FILE}")

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_historical_weekly_summary() -> pd.DataFrame:
    if not HISTORICAL_WEEKLY_FILE.exists():
        raise FileNotFoundError(
            f"Missing historical summary file: {HISTORICAL_WEEKLY_FILE}. "
            "Run ingest_historical.py first."
        )

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

    return df


def get_latest_usage_date(
    historical_df: pd.DataFrame,
    operational_df: pd.DataFrame,
) -> pd.Timestamp:
    latest_historical_date = historical_df["period_end"].max()

    if not operational_df.empty:
        latest_operational_date = operational_df["week_end"].max()
        return max(latest_historical_date, latest_operational_date)

    return latest_historical_date


def calculate_contract_status(
    config: dict,
    historical_df: pd.DataFrame,
    operational_df: pd.DataFrame,
) -> pd.DataFrame:
    contract = config["contract"]
    pricing = config["pricing"]

    contract_start = pd.to_datetime(contract["contract_start_date"])
    contract_end = pd.to_datetime(contract["contract_end_date"])
    purchased_credits = float(contract["purchased_credits"])
    rollover_allowed = bool(contract["rollover_allowed"])

    price_per_credit = float(pricing["current_price_per_credit"])

    historical_contract_df = historical_df[
        (historical_df["period_start"] >= contract_start)
        & (historical_df["period_end"] <= contract_end)
    ]

    historical_credits_used = historical_contract_df["total_credits_used"].sum()

    if operational_df.empty:
        operational_contract_df = operational_df
        operational_credits_used = 0.0
        latest_weekly_burn = 0.0
    else:
        operational_contract_df = operational_df[
            (operational_df["week_start"] >= contract_start)
            & (operational_df["week_end"] <= contract_end)
        ]

        operational_credits_used = operational_contract_df["total_credits_used"].sum()

        if not operational_contract_df.empty:
            latest_weekly_burn = float(
                operational_contract_df.sort_values("week_start")
                .iloc[-1]["total_credits_used"]
            )
        else:
            latest_weekly_burn = 0.0

    total_credits_used = historical_credits_used + operational_credits_used
    credits_remaining = purchased_credits - total_credits_used

    latest_usage_date = get_latest_usage_date(historical_df, operational_df)

    total_contract_days = (contract_end - contract_start).days
    elapsed_days = max((latest_usage_date - contract_start).days, 0)
    remaining_days = max((contract_end - latest_usage_date).days, 0)

    percent_contract_elapsed = (
        elapsed_days / total_contract_days if total_contract_days > 0 else 0
    )

    percent_credits_used = (
        total_credits_used / purchased_credits if purchased_credits > 0 else 0
    )

    burn_pace_ratio = (
        percent_credits_used / percent_contract_elapsed
        if percent_contract_elapsed > 0
        else 0
    )

    weeks_remaining = remaining_days / 7 if remaining_days > 0 else 0

    required_weekly_burn_to_use_all = (
        credits_remaining / weeks_remaining
        if weeks_remaining > 0
        else 0
    )

    projected_cost_used = total_credits_used * price_per_credit
    projected_value_remaining = credits_remaining * price_per_credit

    if burn_pace_ratio < 0.80:
        pacing_status = "UNDERUSING"
    elif burn_pace_ratio <= 1.10:
        pacing_status = "ON_PACE"
    elif burn_pace_ratio <= 1.30:
        pacing_status = "ELEVATED_BURN"
    else:
        pacing_status = "OVERBURNING"

    result = {
        "contract_start_date": contract_start.date(),
        "contract_end_date": contract_end.date(),
        "latest_usage_date": latest_usage_date.date(),
        "purchased_credits": purchased_credits,
        "historical_credits_used": historical_credits_used,
        "operational_credits_used": operational_credits_used,
        "total_credits_used": total_credits_used,
        "credits_remaining": credits_remaining,
        "rollover_allowed": rollover_allowed,
        "price_per_credit": price_per_credit,
        "projected_cost_used": projected_cost_used,
        "projected_value_remaining": projected_value_remaining,
        "total_contract_days": total_contract_days,
        "elapsed_days": elapsed_days,
        "remaining_days": remaining_days,
        "weeks_remaining": weeks_remaining,
        "percent_contract_elapsed": percent_contract_elapsed,
        "percent_credits_used": percent_credits_used,
        "burn_pace_ratio": burn_pace_ratio,
        "required_weekly_burn_to_use_all": required_weekly_burn_to_use_all,
        "latest_weekly_burn": latest_weekly_burn,
        "pacing_status": pacing_status,
    }

    return pd.DataFrame([result])


def print_contract_status(status_df: pd.DataFrame) -> None:
    row = status_df.iloc[0]

    print("\nContract Status")
    print("---------------")
    print(f"Contract period:                 {row['contract_start_date']} to {row['contract_end_date']}")
    print(f"Latest usage date:               {row['latest_usage_date']}")
    print(f"Purchased credits:               {row['purchased_credits']:,.0f}")
    print(f"Historical credits used:         {row['historical_credits_used']:,.2f}")
    print(f"Operational credits used:        {row['operational_credits_used']:,.2f}")
    print(f"Total credits used:              {row['total_credits_used']:,.2f}")
    print(f"Credits remaining:               {row['credits_remaining']:,.2f}")
    print(f"Price per credit:                ${row['price_per_credit']:,.2f}")
    print(f"Estimated value used:            ${row['projected_cost_used']:,.2f}")
    print(f"Estimated value remaining:       ${row['projected_value_remaining']:,.2f}")

    print("\nPacing")
    print("------")
    print(f"Contract elapsed:                {row['percent_contract_elapsed']:.1%}")
    print(f"Credits used:                    {row['percent_credits_used']:.1%}")
    print(f"Burn pace ratio:                 {row['burn_pace_ratio']:.2f}")
    print(f"Remaining days:                  {row['remaining_days']:,.0f}")
    print(f"Weeks remaining:                 {row['weeks_remaining']:,.2f}")
    print(f"Latest weekly burn:              {row['latest_weekly_burn']:,.2f}")
    print(f"Required weekly burn to use all: {row['required_weekly_burn_to_use_all']:,.2f}")
    print(f"Pacing status:                   {row['pacing_status']}")


def main() -> None:
    config = load_config()
    historical_df = load_historical_weekly_summary()
    operational_df = load_operational_weekly_summary()

    status_df = calculate_contract_status(
        config=config,
        historical_df=historical_df,
        operational_df=operational_df,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    status_df.to_csv(OUTPUT_FILE, index=False)

    print_contract_status(status_df)

    print("\nFile written")
    print("------------")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()