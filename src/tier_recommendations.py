from pathlib import Path
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_FILE = PROJECT_ROOT / "config" / "tier_policy_config.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CAP_PRESSURE_USER_SUMMARY_FILE = PROCESSED_DIR / "cap_pressure_history_user_summary.csv"

RECOMMENDATIONS_OUTPUT = PROCESSED_DIR / "tier_recommendations.csv"
RECOMMENDATION_SUMMARY_OUTPUT = PROCESSED_DIR / "tier_recommendation_summary.csv"
TIER_MOVEMENT_SUMMARY_OUTPUT = PROCESSED_DIR / "tier_movement_summary.csv"


ACTION_PRIORITY = {
    "REVIEW_EMERGENCY_OVERRIDE": 1,
    "CONSIDER_MOVE_UP_TIER": 2,
    "CONSIDER_MOVE_DOWN_TIER": 3,
    "MONITOR_RECENT_SPIKE": 4,
    "MONITOR_MORE_HISTORY_NEEDED": 5,
    "NO_CHANGE": 6,
    "RECOMMENDATIONS_DISABLED": 7,
}


def load_tier_config() -> dict:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing tier config file: {CONFIG_FILE}")

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_cap_pressure_user_summary() -> pd.DataFrame:
    if not CAP_PRESSURE_USER_SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Missing cap pressure history summary file: {CAP_PRESSURE_USER_SUMMARY_FILE}. "
            "Run cap_pressure_history.py first."
        )

    df = pd.read_csv(CAP_PRESSURE_USER_SUMMARY_FILE)

    required_columns = [
        "email",
        "weeks_observed",
        "latest_governance_tier",
        "latest_weekly_credit_cap",
        "avg_cap_utilization",
        "latest_cap_utilization",
        "share_weeks_over_90_percent_cap",
        "recommended_action",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in cap pressure history summary: "
            f"{missing}. Re-run cap_pressure_history.py with the latest version."
        )

    numeric_columns = [
        "weeks_observed",
        "latest_weekly_credit_cap",
        "total_credits_used",
        "avg_weekly_credits_used",
        "latest_credits_used",
        "avg_cap_utilization",
        "latest_cap_utilization",
        "max_cap_utilization",
        "weeks_over_90_percent_cap",
        "weeks_at_or_over_cap",
        "share_weeks_over_90_percent_cap",
        "share_weeks_at_or_over_cap",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    date_columns = ["first_week_start", "latest_week_start"]
    for column in date_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    return df


def build_tier_table(tier_config: dict) -> pd.DataFrame:
    tiers = tier_config.get("tiers", {})

    if not tiers:
        raise ValueError("tier_policy_config.yaml must include a non-empty tiers section.")

    tier_rows = []
    for tier_name, values in tiers.items():
        tier_rows.append(
            {
                "tier": tier_name,
                "weekly_credit_cap": float(values["weekly_credit_cap"]),
            }
        )

    tier_table = pd.DataFrame(tier_rows).sort_values("weekly_credit_cap").reset_index(drop=True)

    if "Baseline" not in set(tier_table["tier"]):
        raise ValueError("tier_policy_config.yaml must include a Baseline tier.")

    return tier_table


def get_next_higher_tier(current_tier: str, tier_table: pd.DataFrame) -> tuple[str, float]:
    current = tier_table[tier_table["tier"] == current_tier]

    if current.empty:
        return current_tier, float("nan")

    current_cap = float(current.iloc[0]["weekly_credit_cap"])
    higher = tier_table[tier_table["weekly_credit_cap"] > current_cap]

    if higher.empty:
        return current_tier, current_cap

    row = higher.iloc[0]
    return str(row["tier"]), float(row["weekly_credit_cap"])


def get_next_lower_tier(current_tier: str, tier_table: pd.DataFrame) -> tuple[str, float]:
    current = tier_table[tier_table["tier"] == current_tier]

    if current.empty:
        return current_tier, float("nan")

    current_cap = float(current.iloc[0]["weekly_credit_cap"])
    lower = tier_table[tier_table["weekly_credit_cap"] < current_cap]

    if lower.empty:
        return current_tier, current_cap

    row = lower.iloc[-1]
    return str(row["tier"]), float(row["weekly_credit_cap"])


def estimate_utilization_under_target_cap(avg_weekly_credits: float, target_cap: float) -> float:
    if pd.isna(target_cap) or target_cap <= 0:
        return 0.0
    return avg_weekly_credits / target_cap


def assign_target_tier(row: pd.Series, tier_table: pd.DataFrame) -> tuple[str, float]:
    current_tier = row["latest_governance_tier"]
    current_cap = float(row["latest_weekly_credit_cap"])
    action = row["recommended_action"]

    if action == "CONSIDER_MOVE_UP_TIER":
        return get_next_higher_tier(current_tier, tier_table)

    if action == "CONSIDER_MOVE_DOWN_TIER":
        return get_next_lower_tier(current_tier, tier_table)

    return current_tier, current_cap


def build_reason(row: pd.Series) -> str:
    action = row["recommended_action"]
    weeks = int(row["weeks_observed"])
    avg_utilization = float(row["avg_cap_utilization"])
    latest_utilization = float(row["latest_cap_utilization"])
    share_over_90 = float(row["share_weeks_over_90_percent_cap"])

    if action == "CONSIDER_MOVE_UP_TIER":
        return (
            f"User was at or above 90% utilization in {share_over_90:.0%} "
            f"of observed weeks across {weeks} weeks."
        )

    if action == "CONSIDER_MOVE_DOWN_TIER":
        return (
            f"Average utilization was {avg_utilization:.0%} and latest utilization "
            f"was {latest_utilization:.0%} across {weeks} weeks."
        )

    if action == "MONITOR_RECENT_SPIKE":
        return (
            f"Latest utilization was {latest_utilization:.0%}, but sustained high-pressure "
            "usage has not been observed yet."
        )

    if action == "REVIEW_EMERGENCY_OVERRIDE":
        return "User appeared in an emergency override group and should be reviewed manually."

    if action == "MONITOR_MORE_HISTORY_NEEDED":
        return f"Only {weeks} observed week(s) are available; more history is needed."

    if action == "RECOMMENDATIONS_DISABLED":
        return "Recommendation logic is disabled in tier_policy_config.yaml."

    return "No tier movement recommended based on current policy thresholds."


def assign_review_priority(action: str) -> str:
    if action == "REVIEW_EMERGENCY_OVERRIDE":
        return "URGENT"
    if action in {"CONSIDER_MOVE_UP_TIER", "CONSIDER_MOVE_DOWN_TIER"}:
        return "ACTIONABLE"
    if action in {"MONITOR_RECENT_SPIKE", "MONITOR_MORE_HISTORY_NEEDED"}:
        return "MONITOR"
    return "INFORMATIONAL"


def build_tier_recommendations(
    user_summary: pd.DataFrame,
    tier_table: pd.DataFrame,
) -> pd.DataFrame:
    recommendations = user_summary.copy()

    target_values = recommendations.apply(
        lambda row: assign_target_tier(row, tier_table),
        axis=1,
    )

    recommendations["recommended_tier"] = [value[0] for value in target_values]
    recommendations["recommended_weekly_credit_cap"] = [value[1] for value in target_values]

    recommendations["recommended_cap_change"] = (
        recommendations["recommended_weekly_credit_cap"]
        - recommendations["latest_weekly_credit_cap"]
    )

    if "avg_weekly_credits_used" in recommendations.columns:
        recommendations["estimated_avg_utilization_after_change"] = recommendations.apply(
            lambda row: estimate_utilization_under_target_cap(
                avg_weekly_credits=float(row["avg_weekly_credits_used"]),
                target_cap=float(row["recommended_weekly_credit_cap"]),
            ),
            axis=1,
        )
    else:
        recommendations["estimated_avg_utilization_after_change"] = 0.0

    recommendations["recommendation_reason"] = recommendations.apply(build_reason, axis=1)
    recommendations["review_priority"] = recommendations["recommended_action"].apply(assign_review_priority)
    recommendations["action_priority_rank"] = recommendations["recommended_action"].map(
        ACTION_PRIORITY
    ).fillna(99)

    preferred_columns = [
        "email",
        "latest_name",
        "latest_department",
        "latest_governance_tier",
        "latest_weekly_credit_cap",
        "recommended_action",
        "recommended_tier",
        "recommended_weekly_credit_cap",
        "recommended_cap_change",
        "review_priority",
        "recommendation_reason",
        "weeks_observed",
        "first_week_start",
        "latest_week_start",
        "total_credits_used",
        "avg_weekly_credits_used",
        "latest_credits_used",
        "avg_cap_utilization",
        "latest_cap_utilization",
        "max_cap_utilization",
        "estimated_avg_utilization_after_change",
        "weeks_over_90_percent_cap",
        "weeks_at_or_over_cap",
        "share_weeks_over_90_percent_cap",
        "share_weeks_at_or_over_cap",
        "pressure_trend",
        "ever_emergency_override",
        "emergency_override_weeks",
        "most_common_governance_tier",
    ]

    output_columns = [col for col in preferred_columns if col in recommendations.columns]

    recommendations = recommendations.sort_values(
        ["action_priority_rank", "latest_cap_utilization", "total_credits_used"],
        ascending=[True, False, False],
    )

    return recommendations[output_columns]


def build_recommendation_summary(recommendations: pd.DataFrame) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame()

    summary = (
        recommendations.groupby(["recommended_action", "review_priority"], as_index=False)
        .agg(
            users=("email", "count"),
            total_credits_used=("total_credits_used", "sum")
            if "total_credits_used" in recommendations.columns
            else ("email", "count"),
            avg_latest_utilization=("latest_cap_utilization", "mean"),
            avg_historical_utilization=("avg_cap_utilization", "mean"),
            total_recommended_cap_change=("recommended_cap_change", "sum"),
        )
        .sort_values("users", ascending=False)
    )

    total_users = summary["users"].sum()
    summary["share_of_users"] = summary["users"] / total_users if total_users > 0 else 0

    return summary


def build_tier_movement_summary(recommendations: pd.DataFrame) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame()

    actionable = recommendations[
        recommendations["latest_governance_tier"] != recommendations["recommended_tier"]
    ].copy()

    if actionable.empty:
        return pd.DataFrame(
            columns=[
                "latest_governance_tier",
                "recommended_tier",
                "users",
                "total_recommended_cap_change",
                "avg_latest_utilization",
                "avg_historical_utilization",
            ]
        )

    movement_summary = (
        actionable.groupby(["latest_governance_tier", "recommended_tier"], as_index=False)
        .agg(
            users=("email", "count"),
            total_recommended_cap_change=("recommended_cap_change", "sum"),
            avg_latest_utilization=("latest_cap_utilization", "mean"),
            avg_historical_utilization=("avg_cap_utilization", "mean"),
        )
        .sort_values("users", ascending=False)
    )

    return movement_summary


def print_summary(
    recommendations: pd.DataFrame,
    recommendation_summary: pd.DataFrame,
    movement_summary: pd.DataFrame,
) -> None:
    print("\nTier Recommendations Complete")
    print("-----------------------------")

    if recommendations.empty:
        print("No user recommendation rows found.")
        return

    total_users = recommendations["email"].nunique()
    actionable_users = int(
        recommendations["recommended_action"].isin(
            ["CONSIDER_MOVE_UP_TIER", "CONSIDER_MOVE_DOWN_TIER", "REVIEW_EMERGENCY_OVERRIDE"]
        ).sum()
    )

    print(f"Users evaluated:                 {total_users:,.0f}")
    print(f"Actionable review items:         {actionable_users:,.0f}")

    print("\nRecommendation Summary")
    print("----------------------")
    for _, row in recommendation_summary.iterrows():
        print(
            f"{row['recommended_action']}: "
            f"{row['users']:,.0f} users, "
            f"{row['avg_latest_utilization']:.1%} avg latest utilization"
        )

    if not movement_summary.empty:
        print("\nTier Movement Summary")
        print("---------------------")
        for _, row in movement_summary.iterrows():
            print(
                f"{row['latest_governance_tier']} -> {row['recommended_tier']}: "
                f"{row['users']:,.0f} users, "
                f"{row['total_recommended_cap_change']:,.0f} net weekly credits"
            )

    print("\nFiles written")
    print("-------------")
    print(RECOMMENDATIONS_OUTPUT)
    print(RECOMMENDATION_SUMMARY_OUTPUT)
    print(TIER_MOVEMENT_SUMMARY_OUTPUT)


def main() -> None:
    tier_config = load_tier_config()
    tier_table = build_tier_table(tier_config)
    user_summary = load_cap_pressure_user_summary()

    recommendations = build_tier_recommendations(
        user_summary=user_summary,
        tier_table=tier_table,
    )

    recommendation_summary = build_recommendation_summary(recommendations)
    movement_summary = build_tier_movement_summary(recommendations)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(RECOMMENDATIONS_OUTPUT, index=False)
    recommendation_summary.to_csv(RECOMMENDATION_SUMMARY_OUTPUT, index=False)
    movement_summary.to_csv(TIER_MOVEMENT_SUMMARY_OUTPUT, index=False)

    print_summary(
        recommendations=recommendations,
        recommendation_summary=recommendation_summary,
        movement_summary=movement_summary,
    )


if __name__ == "__main__":
    main()
