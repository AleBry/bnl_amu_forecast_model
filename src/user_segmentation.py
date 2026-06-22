from pathlib import Path
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_FILE = PROJECT_ROOT / "config" / "tier_policy_config.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

HISTORICAL_USAGE_FILE = PROCESSED_DIR / "historical_usage_cleaned.csv"

USER_SEGMENT_OUTPUT = PROCESSED_DIR / "historical_user_segments.csv"
SEGMENT_SUMMARY_OUTPUT = PROCESSED_DIR / "historical_segment_summary.csv"


def load_tier_config() -> dict:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing tier config file: {CONFIG_FILE}")

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_historical_usage() -> pd.DataFrame:
    if not HISTORICAL_USAGE_FILE.exists():
        raise FileNotFoundError(
            f"Missing historical usage file: {HISTORICAL_USAGE_FILE}. "
            "Run ingest_historical.py first."
        )

    df = pd.read_csv(HISTORICAL_USAGE_FILE)
    df["period_start"] = pd.to_datetime(df["period_start"])
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["credits_used"] = pd.to_numeric(df["credits_used"], errors="coerce").fillna(0)

    return df


def build_segment_thresholds(tier_config: dict) -> list[dict]:
    tiers = tier_config["tiers"]

    tier_rows = []

    for tier_name, values in tiers.items():
        cap = float(values["weekly_credit_cap"])
        tier_rows.append(
            {
                "tier_name": tier_name,
                "weekly_credit_cap": cap,
            }
        )

    tier_rows = sorted(tier_rows, key=lambda x: x["weekly_credit_cap"])

    thresholds = [
        {
            "segment": "Inactive",
            "min_weekly_credits": 0,
            "max_weekly_credits": 0,
        }
    ]

    previous_cap = 0

    for row in tier_rows:
        tier_name = row["tier_name"]
        cap = row["weekly_credit_cap"]

        thresholds.append(
            {
                "segment": f"Up to {tier_name}",
                "min_weekly_credits": previous_cap,
                "max_weekly_credits": cap,
            }
        )

        previous_cap = cap

    highest_cap = tier_rows[-1]["weekly_credit_cap"]

    thresholds.append(
        {
            "segment": f"Above {tier_rows[-1]['tier_name']}",
            "min_weekly_credits": highest_cap,
            "max_weekly_credits": None,
        }
    )

    return thresholds


def assign_segment(avg_weekly_credits: float, thresholds: list[dict]) -> str:
    if avg_weekly_credits <= 0:
        return "Inactive"

    for rule in thresholds:
        min_value = rule["min_weekly_credits"]
        max_value = rule["max_weekly_credits"]

        if rule["segment"] == "Inactive":
            continue

        if max_value is None:
            if avg_weekly_credits > min_value:
                return rule["segment"]

        else:
            if avg_weekly_credits > min_value and avg_weekly_credits <= max_value:
                return rule["segment"]

    return "Unclassified"


def build_user_segments(
    historical_df: pd.DataFrame,
    thresholds: list[dict],
) -> pd.DataFrame:
    total_weeks = historical_df["period_start"].nunique()

    user_segments = (
        historical_df.groupby("email", as_index=False)
        .agg(
            total_credits=("credits_used", "sum"),
            avg_weekly_credits=("credits_used", "mean"),
            median_weekly_credits=("credits_used", "median"),
            max_weekly_credits=("credits_used", "max"),
            active_credit_weeks=("is_credit_active", "sum"),
            total_messages=("messages", "sum"),
            active_message_weeks=("is_message_active", "sum"),
            first_seen=("period_start", "min"),
            last_seen=("period_start", "max"),
        )
    )

    user_segments["historical_weeks_observed"] = total_weeks

    user_segments["credit_active_rate"] = (
        user_segments["active_credit_weeks"] / total_weeks
    )

    user_segments["message_active_rate"] = (
        user_segments["active_message_weeks"] / total_weeks
    )

    user_segments["usage_segment"] = user_segments["avg_weekly_credits"].apply(
        lambda value: assign_segment(value, thresholds)
    )

    user_segments = user_segments.sort_values(
        "total_credits",
        ascending=False,
    )

    return user_segments


def build_segment_summary(user_segments: pd.DataFrame) -> pd.DataFrame:
    summary = (
        user_segments.groupby("usage_segment", as_index=False)
        .agg(
            users=("email", "count"),
            total_credits=("total_credits", "sum"),
            avg_user_total_credits=("total_credits", "mean"),
            avg_weekly_credits=("avg_weekly_credits", "mean"),
            median_weekly_credits=("median_weekly_credits", "median"),
            avg_credit_active_rate=("credit_active_rate", "mean"),
        )
    )

    total_credits = summary["total_credits"].sum()

    summary["share_of_total_credits"] = (
        summary["total_credits"] / total_credits
        if total_credits > 0
        else 0
    )

    return summary.sort_values("avg_weekly_credits")


def print_summary(user_segments: pd.DataFrame, segment_summary: pd.DataFrame) -> None:
    print("\nUser Segmentation Complete")
    print("--------------------------")
    print(f"Users segmented: {len(user_segments):,}")

    print("\nSegment Summary")
    print("---------------")

    for _, row in segment_summary.iterrows():
        print(
            f"{row['usage_segment']}: "
            f"{row['users']:,.0f} users, "
            f"{row['total_credits']:,.2f} credits, "
            f"{row['share_of_total_credits']:.1%} of total"
        )

    print("\nFiles written")
    print("-------------")
    print(USER_SEGMENT_OUTPUT)
    print(SEGMENT_SUMMARY_OUTPUT)


def main() -> None:
    tier_config = load_tier_config()
    historical_df = load_historical_usage()

    thresholds = build_segment_thresholds(tier_config)

    user_segments = build_user_segments(
        historical_df=historical_df,
        thresholds=thresholds,
    )

    segment_summary = build_segment_summary(user_segments)

    user_segments.to_csv(USER_SEGMENT_OUTPUT, index=False)
    segment_summary.to_csv(SEGMENT_SUMMARY_OUTPUT, index=False)

    print_summary(user_segments, segment_summary)


if __name__ == "__main__":
    main()