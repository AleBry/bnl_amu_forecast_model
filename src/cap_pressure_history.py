from pathlib import Path
import ast
import math
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_FILE = PROJECT_ROOT / "config" / "tier_policy_config.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OPERATIONAL_HISTORY_FILE = PROCESSED_DIR / "weekly_operational_usage_all.csv"

USER_WEEK_OUTPUT = PROCESSED_DIR / "cap_pressure_history_user_week.csv"
USER_SUMMARY_OUTPUT = PROCESSED_DIR / "cap_pressure_history_user_summary.csv"
TIER_WEEK_OUTPUT = PROCESSED_DIR / "cap_pressure_history_tier_week.csv"
TIER_SUMMARY_OUTPUT = PROCESSED_DIR / "cap_pressure_history_tier_summary.csv"


DEFAULT_GROUP_TO_TIER = {
    "Advanced Credit Users": "Advanced",
    "High Credit Consumption Users": "Super",
    "One K Credit Users": "Highest",
}

DEFAULT_SPECIAL_GROUPS = {
    "Emergency Credit Users": "emergency_override",
}


UP_RECOMMENDATION_MIN_WEEKS = 3
UP_RECOMMENDATION_MIN_90_SHARE = 0.50
DOWN_RECOMMENDATION_MIN_WEEKS = 4
DOWN_RECOMMENDATION_MAX_AVG_UTILIZATION = 0.25
SPIKE_MONITOR_THRESHOLD = 0.90


def load_tier_config() -> dict:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing tier config file: {CONFIG_FILE}")

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_operational_history() -> pd.DataFrame:
    if not OPERATIONAL_HISTORY_FILE.exists():
        raise FileNotFoundError(
            f"Missing operational history file: {OPERATIONAL_HISTORY_FILE}. "
            "Run ingest_weekly.py first."
        )

    df = pd.read_csv(OPERATIONAL_HISTORY_FILE)

    required_columns = ["email", "credits_used", "groups", "week_start", "week_end"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in operational history file: {missing}")

    df["credits_used"] = pd.to_numeric(df["credits_used"], errors="coerce").fillna(0)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["week_end"] = pd.to_datetime(df["week_end"])

    if "messages" in df.columns:
        df["messages"] = pd.to_numeric(df["messages"], errors="coerce").fillna(0)

    return df.sort_values(["week_start", "email"])


def build_tier_cap_lookup(tier_config: dict) -> dict[str, float]:
    tiers = tier_config["tiers"]

    return {
        tier_name: float(values["weekly_credit_cap"])
        for tier_name, values in tiers.items()
    }


def get_group_to_tier_mapping(tier_config: dict) -> dict:
    return tier_config.get("group_to_tier", DEFAULT_GROUP_TO_TIER)


def get_special_groups(tier_config: dict) -> dict:
    return tier_config.get("special_groups", DEFAULT_SPECIAL_GROUPS)


def parse_groups(groups_value) -> list[str]:
    if pd.isna(groups_value):
        return []

    if isinstance(groups_value, dict):
        return [str(value) for value in groups_value.values()]

    if isinstance(groups_value, list):
        return [str(value) for value in groups_value]

    text = str(groups_value).strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]

    if isinstance(parsed, dict):
        return [str(value) for value in parsed.values()]

    if isinstance(parsed, list):
        return [str(value) for value in parsed]

    return [str(parsed)]


def assign_governance_tier(
    group_names: list[str],
    group_to_tier: dict,
    tier_caps: dict[str, float],
) -> str:
    matched_tiers = [
        group_to_tier[group]
        for group in group_names
        if group in group_to_tier
    ]

    if not matched_tiers:
        return "Baseline"

    return max(
        matched_tiers,
        key=lambda tier: tier_caps.get(tier, -math.inf),
    )


def identify_special_group_flags(
    group_names: list[str],
    special_groups: dict,
) -> dict:
    flags = {
        flag_name: False
        for flag_name in special_groups.values()
    }

    for group in group_names:
        if group in special_groups:
            flags[special_groups[group]] = True

    return flags


def assign_pressure_flag(cap_utilization: float) -> str:
    if cap_utilization >= 1.10:
        return "ABOVE_CAP_110_PLUS"
    if cap_utilization >= 1.00:
        return "AT_OR_ABOVE_CAP"
    if cap_utilization >= 0.90:
        return "HIGH_PRESSURE_90_PLUS"
    if cap_utilization >= 0.80:
        return "ELEVATED_PRESSURE_80_PLUS"
    return "NORMAL"


def validate_tier_config(tier_config: dict) -> None:
    tier_caps = build_tier_cap_lookup(tier_config)
    group_to_tier = get_group_to_tier_mapping(tier_config)

    if "Baseline" not in tier_caps:
        raise ValueError("tier_policy_config.yaml must include a Baseline tier.")

    invalid_mapped_tiers = sorted(
        {
            tier_name
            for tier_name in group_to_tier.values()
            if tier_name not in tier_caps
        }
    )

    if invalid_mapped_tiers:
        raise ValueError(
            "group_to_tier contains tier names not present in tiers: "
            f"{invalid_mapped_tiers}"
        )


def build_user_week_history(
    operational_df: pd.DataFrame,
    tier_config: dict,
) -> pd.DataFrame:
    validate_tier_config(tier_config)

    tier_caps = build_tier_cap_lookup(tier_config)
    group_to_tier = get_group_to_tier_mapping(tier_config)
    special_groups = get_special_groups(tier_config)

    df = operational_df.copy()
    df["parsed_groups"] = df["groups"].apply(parse_groups)

    df["governance_tier"] = df["parsed_groups"].apply(
        lambda groups: assign_governance_tier(
            group_names=groups,
            group_to_tier=group_to_tier,
            tier_caps=tier_caps,
        )
    )

    df["weekly_credit_cap"] = df["governance_tier"].map(tier_caps)
    df["cap_utilization"] = df["credits_used"] / df["weekly_credit_cap"]
    df["remaining_weekly_credits"] = df["weekly_credit_cap"] - df["credits_used"]

    df["is_over_80_percent_cap"] = df["cap_utilization"] >= 0.80
    df["is_over_90_percent_cap"] = df["cap_utilization"] >= 0.90
    df["is_at_or_over_cap"] = df["cap_utilization"] >= 1.00
    df["is_over_110_percent_cap"] = df["cap_utilization"] >= 1.10
    df["pressure_flag"] = df["cap_utilization"].apply(assign_pressure_flag)

    special_flags = df["parsed_groups"].apply(
        lambda groups: identify_special_group_flags(groups, special_groups)
    )

    if len(special_flags) > 0:
        special_flags_df = pd.DataFrame(list(special_flags)).fillna(False)
        df = pd.concat([df.reset_index(drop=True), special_flags_df.reset_index(drop=True)], axis=1)

    df["groups_parsed_text"] = df["parsed_groups"].apply(lambda groups: "; ".join(groups))

    preferred_columns = [
        "week_start",
        "week_end",
        "email",
        "name",
        "department",
        "governance_tier",
        "weekly_credit_cap",
        "credits_used",
        "cap_utilization",
        "remaining_weekly_credits",
        "pressure_flag",
        "is_over_80_percent_cap",
        "is_over_90_percent_cap",
        "is_at_or_over_cap",
        "is_over_110_percent_cap",
        "groups_parsed_text",
    ]

    special_flag_columns = [
        column
        for column in special_groups.values()
        if column in df.columns
    ]

    optional_columns = [
        "messages",
        "is_credit_active",
        "is_message_active",
        "user_status",
        "role",
        "user_role",
        "source_file",
    ]

    output_columns = [
        column
        for column in preferred_columns + special_flag_columns + optional_columns
        if column in df.columns
    ]

    return df[output_columns].sort_values(["week_start", "credits_used"], ascending=[True, False])


def latest_value(series: pd.Series):
    return series.iloc[-1]


def most_common_value(series: pd.Series):
    modes = series.dropna().mode()
    if len(modes) == 0:
        return None
    return modes.iloc[0]


def assign_trend_label(first_utilization: float, latest_utilization: float) -> str:
    change = latest_utilization - first_utilization

    if change >= 0.20:
        return "INCREASING_PRESSURE"
    if change <= -0.20:
        return "DECREASING_PRESSURE"
    return "STABLE_PRESSURE"


def assign_tier_recommendation(row: pd.Series) -> str:
    if bool(row.get("ever_emergency_override", False)):
        return "REVIEW_EMERGENCY_OVERRIDE"

    if row["weeks_observed"] < 2:
        return "MONITOR_MORE_HISTORY_NEEDED"

    if (
        row["weeks_observed"] >= UP_RECOMMENDATION_MIN_WEEKS
        and row["share_weeks_over_90_percent_cap"] >= UP_RECOMMENDATION_MIN_90_SHARE
    ):
        return "CONSIDER_MOVE_UP_TIER"

    if (
        row["weeks_observed"] >= DOWN_RECOMMENDATION_MIN_WEEKS
        and row["avg_cap_utilization"] <= DOWN_RECOMMENDATION_MAX_AVG_UTILIZATION
        and row["latest_cap_utilization"] <= DOWN_RECOMMENDATION_MAX_AVG_UTILIZATION
    ):
        return "CONSIDER_MOVE_DOWN_TIER"

    if (
        row["latest_cap_utilization"] >= SPIKE_MONITOR_THRESHOLD
        and row["share_weeks_over_90_percent_cap"] < UP_RECOMMENDATION_MIN_90_SHARE
    ):
        return "MONITOR_RECENT_SPIKE"

    return "NO_CHANGE"


def build_user_summary(user_week_history: pd.DataFrame) -> pd.DataFrame:
    if user_week_history.empty:
        return pd.DataFrame()

    df = user_week_history.sort_values(["email", "week_start"])

    aggregation = {
        "weeks_observed": ("week_start", "nunique"),
        "first_week_start": ("week_start", "min"),
        "latest_week_start": ("week_start", "max"),
        "total_credits_used": ("credits_used", "sum"),
        "avg_weekly_credits_used": ("credits_used", "mean"),
        "median_weekly_credits_used": ("credits_used", "median"),
        "max_weekly_credits_used": ("credits_used", "max"),
        "avg_cap_utilization": ("cap_utilization", "mean"),
        "median_cap_utilization": ("cap_utilization", "median"),
        "max_cap_utilization": ("cap_utilization", "max"),
        "weeks_over_80_percent_cap": ("is_over_80_percent_cap", "sum"),
        "weeks_over_90_percent_cap": ("is_over_90_percent_cap", "sum"),
        "weeks_at_or_over_cap": ("is_at_or_over_cap", "sum"),
        "weeks_over_110_percent_cap": ("is_over_110_percent_cap", "sum"),
        "latest_governance_tier": ("governance_tier", latest_value),
        "most_common_governance_tier": ("governance_tier", most_common_value),
        "latest_weekly_credit_cap": ("weekly_credit_cap", latest_value),
        "latest_credits_used": ("credits_used", latest_value),
        "latest_cap_utilization": ("cap_utilization", latest_value),
        "first_cap_utilization": ("cap_utilization", lambda s: s.iloc[0]),
    }

    optional_first_last_columns = ["name", "department"]
    for column in optional_first_last_columns:
        if column in df.columns:
            aggregation[f"latest_{column}"] = (column, latest_value)

    if "messages" in df.columns:
        aggregation["total_messages"] = ("messages", "sum")
        aggregation["avg_weekly_messages"] = ("messages", "mean")

    if "emergency_override" in df.columns:
        aggregation["ever_emergency_override"] = ("emergency_override", "max")
        aggregation["emergency_override_weeks"] = ("emergency_override", "sum")

    user_summary = df.groupby("email", as_index=False).agg(**aggregation)

    user_summary["share_weeks_over_80_percent_cap"] = (
        user_summary["weeks_over_80_percent_cap"] / user_summary["weeks_observed"]
    )
    user_summary["share_weeks_over_90_percent_cap"] = (
        user_summary["weeks_over_90_percent_cap"] / user_summary["weeks_observed"]
    )
    user_summary["share_weeks_at_or_over_cap"] = (
        user_summary["weeks_at_or_over_cap"] / user_summary["weeks_observed"]
    )

    if "ever_emergency_override" not in user_summary.columns:
        user_summary["ever_emergency_override"] = False
        user_summary["emergency_override_weeks"] = 0

    user_summary["pressure_trend"] = user_summary.apply(
        lambda row: assign_trend_label(
            row["first_cap_utilization"],
            row["latest_cap_utilization"],
        ),
        axis=1,
    )

    user_summary["recommended_action"] = user_summary.apply(
        assign_tier_recommendation,
        axis=1,
    )

    return user_summary.sort_values(
        ["recommended_action", "latest_cap_utilization", "total_credits_used"],
        ascending=[True, False, False],
    )


def build_tier_week_summary(user_week_history: pd.DataFrame) -> pd.DataFrame:
    if user_week_history.empty:
        return pd.DataFrame()

    tier_week = (
        user_week_history.groupby(["week_start", "week_end", "governance_tier"], as_index=False)
        .agg(
            users=("email", "count"),
            credit_active_users=("credits_used", lambda s: int((s > 0).sum())),
            total_credits_used=("credits_used", "sum"),
            avg_credits_used=("credits_used", "mean"),
            median_credits_used=("credits_used", "median"),
            avg_cap_utilization=("cap_utilization", "mean"),
            median_cap_utilization=("cap_utilization", "median"),
            users_over_80_percent_cap=("is_over_80_percent_cap", "sum"),
            users_over_90_percent_cap=("is_over_90_percent_cap", "sum"),
            users_at_or_over_cap=("is_at_or_over_cap", "sum"),
            users_over_110_percent_cap=("is_over_110_percent_cap", "sum"),
        )
        .sort_values(["week_start", "governance_tier"])
    )

    return tier_week


def build_tier_summary(user_week_history: pd.DataFrame) -> pd.DataFrame:
    if user_week_history.empty:
        return pd.DataFrame()

    tier_summary = (
        user_week_history.groupby("governance_tier", as_index=False)
        .agg(
            user_week_rows=("email", "count"),
            unique_users=("email", "nunique"),
            weeks_observed=("week_start", "nunique"),
            total_credits_used=("credits_used", "sum"),
            avg_weekly_credits_used=("credits_used", "mean"),
            median_weekly_credits_used=("credits_used", "median"),
            avg_cap_utilization=("cap_utilization", "mean"),
            median_cap_utilization=("cap_utilization", "median"),
            user_weeks_over_80_percent_cap=("is_over_80_percent_cap", "sum"),
            user_weeks_over_90_percent_cap=("is_over_90_percent_cap", "sum"),
            user_weeks_at_or_over_cap=("is_at_or_over_cap", "sum"),
            user_weeks_over_110_percent_cap=("is_over_110_percent_cap", "sum"),
        )
    )

    total_credits = tier_summary["total_credits_used"].sum()
    tier_summary["share_of_total_credits"] = (
        tier_summary["total_credits_used"] / total_credits
        if total_credits > 0
        else 0
    )

    tier_summary["share_user_weeks_over_90_percent_cap"] = (
        tier_summary["user_weeks_over_90_percent_cap"] / tier_summary["user_week_rows"]
    )

    return tier_summary.sort_values("total_credits_used", ascending=False)


def print_summary(
    user_week_history: pd.DataFrame,
    user_summary: pd.DataFrame,
    tier_summary: pd.DataFrame,
) -> None:
    print("\nCap Pressure History Complete")
    print("-----------------------------")

    if user_week_history.empty:
        print("No usage history rows found.")
        return

    first_week = user_week_history["week_start"].min().date()
    latest_week = user_week_history["week_start"].max().date()
    weeks = user_week_history["week_start"].nunique()
    users = user_week_history["email"].nunique()
    total_credits = user_week_history["credits_used"].sum()

    print(f"History range:                  {first_week} to {latest_week}")
    print(f"Weeks observed:                 {weeks:,.0f}")
    print(f"Unique users observed:          {users:,.0f}")
    print(f"User-week rows:                 {len(user_week_history):,.0f}")
    print(f"Total credits used:             {total_credits:,.2f}")

    print("\nGovernance Recommendations")
    print("--------------------------")
    recommendation_counts = user_summary["recommended_action"].value_counts()
    for action, count in recommendation_counts.items():
        print(f"{action}: {count:,.0f} users")

    if not tier_summary.empty:
        print("\nTier History Summary")
        print("--------------------")
        for _, row in tier_summary.iterrows():
            print(
                f"{row['governance_tier']}: "
                f"{row['unique_users']:,.0f} unique users, "
                f"{row['total_credits_used']:,.2f} credits, "
                f"{row['avg_cap_utilization']:.1%} avg utilization"
            )

    print("\nFiles written")
    print("-------------")
    print(USER_WEEK_OUTPUT)
    print(USER_SUMMARY_OUTPUT)
    print(TIER_WEEK_OUTPUT)
    print(TIER_SUMMARY_OUTPUT)


def main() -> None:
    tier_config = load_tier_config()
    operational_df = load_operational_history()

    user_week_history = build_user_week_history(
        operational_df=operational_df,
        tier_config=tier_config,
    )

    user_summary = build_user_summary(user_week_history)
    tier_week_summary = build_tier_week_summary(user_week_history)
    tier_summary = build_tier_summary(user_week_history)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    user_week_history.to_csv(USER_WEEK_OUTPUT, index=False)
    user_summary.to_csv(USER_SUMMARY_OUTPUT, index=False)
    tier_week_summary.to_csv(TIER_WEEK_OUTPUT, index=False)
    tier_summary.to_csv(TIER_SUMMARY_OUTPUT, index=False)

    print_summary(
        user_week_history=user_week_history,
        user_summary=user_summary,
        tier_summary=tier_summary,
    )


if __name__ == "__main__":
    main()
