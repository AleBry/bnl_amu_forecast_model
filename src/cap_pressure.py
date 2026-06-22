from pathlib import Path
import ast
import math
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_FILE = PROJECT_ROOT / "config" / "tier_policy_config.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LATEST_WEEK_FILE = PROCESSED_DIR / "latest_week_operational_usage_cleaned.csv"

USER_DETAIL_OUTPUT = PROCESSED_DIR / "cap_pressure_user_detail.csv"
SUMMARY_OUTPUT = PROCESSED_DIR / "cap_pressure_summary.csv"


DEFAULT_GROUP_TO_TIER = {
    "Advanced Credit Users": "Advanced",
    "High Credit Consumption Users": "Super",
    "One K Credit Users": "Highest",
}

DEFAULT_SPECIAL_GROUPS = {
    "Emergency Credit Users": "emergency_override",
}


def load_tier_config() -> dict:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing tier config file: {CONFIG_FILE}")

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_latest_week_usage() -> pd.DataFrame:
    if not LATEST_WEEK_FILE.exists():
        raise FileNotFoundError(
            f"Missing latest weekly usage file: {LATEST_WEEK_FILE}. "
            "Run ingest_weekly.py first."
        )

    df = pd.read_csv(LATEST_WEEK_FILE)

    required_columns = ["email", "credits_used", "groups", "week_start", "week_end"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in latest weekly file: {missing}")

    df["credits_used"] = pd.to_numeric(df["credits_used"], errors="coerce").fillna(0)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["week_end"] = pd.to_datetime(df["week_end"])

    return df


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
        return list(groups_value.values())

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


def calculate_top_share(usage_values: pd.Series, top_fraction: float) -> float:
    usage = usage_values.fillna(0).clip(lower=0).sort_values(ascending=False)
    total_usage = usage.sum()

    if total_usage <= 0 or len(usage) == 0:
        return 0.0

    top_n = max(1, math.ceil(len(usage) * top_fraction))
    return float(usage.head(top_n).sum() / total_usage)


def calculate_gini(usage_values: pd.Series) -> float:
    usage = usage_values.fillna(0).clip(lower=0).sort_values().reset_index(drop=True)
    n = len(usage)
    total_usage = usage.sum()

    if n == 0 or total_usage <= 0:
        return 0.0

    weighted_sum = sum((index + 1) * value for index, value in enumerate(usage))
    gini = (2 * weighted_sum) / (n * total_usage) - (n + 1) / n

    return float(gini)


def calculate_hhi(usage_values: pd.Series) -> float:
    usage = usage_values.fillna(0).clip(lower=0)
    total_usage = usage.sum()

    if total_usage <= 0:
        return 0.0

    shares = usage / total_usage
    return float((shares ** 2).sum())


def calculate_cap_pressure_index(user_detail: pd.DataFrame) -> float:
    if user_detail.empty:
        return 0.0

    avg_utilization = user_detail["cap_utilization"].clip(lower=0, upper=1).mean()

    users = len(user_detail)
    share_80_plus = (user_detail["cap_utilization"] >= 0.80).sum() / users
    share_90_plus = (user_detail["cap_utilization"] >= 0.90).sum() / users
    share_100_plus = (user_detail["cap_utilization"] >= 1.00).sum() / users

    threshold_pressure = (
        0.25 * share_80_plus
        + 0.35 * share_90_plus
        + 0.40 * share_100_plus
    )

    top_5_share = calculate_top_share(user_detail["credits_used"], 0.05)
    top_10_share = calculate_top_share(user_detail["credits_used"], 0.10)
    gini = calculate_gini(user_detail["credits_used"])

    concentration_pressure = (
        0.40 * min(top_5_share / 0.50, 1)
        + 0.30 * min(top_10_share / 0.70, 1)
        + 0.30 * min(gini / 0.80, 1)
    )

    cap_pressure_index = 100 * (
        0.40 * avg_utilization
        + 0.30 * threshold_pressure
        + 0.30 * concentration_pressure
    )

    return float(min(max(cap_pressure_index, 0), 100))


def assign_pressure_level(cap_pressure_index: float) -> str:
    if cap_pressure_index >= 80:
        return "CRITICAL"
    if cap_pressure_index >= 60:
        return "HIGH"
    if cap_pressure_index >= 30:
        return "MODERATE"
    return "LOW"


def build_user_detail(
    weekly_df: pd.DataFrame,
    tier_config: dict,
) -> pd.DataFrame:
    tier_caps = build_tier_cap_lookup(tier_config)
    group_to_tier = get_group_to_tier_mapping(tier_config)
    special_groups = get_special_groups(tier_config)

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

    df = weekly_df.copy()
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

    return df[output_columns].sort_values("credits_used", ascending=False)


def build_summary(user_detail: pd.DataFrame) -> pd.DataFrame:
    users = len(user_detail)
    total_credits = float(user_detail["credits_used"].sum()) if users > 0 else 0.0
    credit_active_users = int((user_detail["credits_used"] > 0).sum()) if users > 0 else 0

    cap_pressure_index = calculate_cap_pressure_index(user_detail)
    pressure_level = assign_pressure_level(cap_pressure_index)

    week_start = user_detail["week_start"].iloc[0] if users > 0 else None
    week_end = user_detail["week_end"].iloc[0] if users > 0 else None

    summary = {
        "week_start": week_start,
        "week_end": week_end,
        "users": users,
        "credit_active_users": credit_active_users,
        "total_credits_used": total_credits,
        "avg_credits_per_user": float(user_detail["credits_used"].mean()) if users > 0 else 0.0,
        "median_credits_per_user": float(user_detail["credits_used"].median()) if users > 0 else 0.0,
        "p95_credits_per_user": float(user_detail["credits_used"].quantile(0.95)) if users > 0 else 0.0,
        "avg_cap_utilization": float(user_detail["cap_utilization"].mean()) if users > 0 else 0.0,
        "median_cap_utilization": float(user_detail["cap_utilization"].median()) if users > 0 else 0.0,
        "users_over_80_percent_cap": int(user_detail["is_over_80_percent_cap"].sum()) if users > 0 else 0,
        "users_over_90_percent_cap": int(user_detail["is_over_90_percent_cap"].sum()) if users > 0 else 0,
        "users_at_or_over_cap": int(user_detail["is_at_or_over_cap"].sum()) if users > 0 else 0,
        "users_over_110_percent_cap": int(user_detail["is_over_110_percent_cap"].sum()) if users > 0 else 0,
        "top_1_percent_consumption_share": calculate_top_share(user_detail["credits_used"], 0.01),
        "top_5_percent_consumption_share": calculate_top_share(user_detail["credits_used"], 0.05),
        "top_10_percent_consumption_share": calculate_top_share(user_detail["credits_used"], 0.10),
        "gini_coefficient": calculate_gini(user_detail["credits_used"]),
        "hhi": calculate_hhi(user_detail["credits_used"]),
        "cap_pressure_index": cap_pressure_index,
        "pressure_level": pressure_level,
    }

    if "emergency_override" in user_detail.columns:
        summary["emergency_override_users"] = int(user_detail["emergency_override"].sum())

    return pd.DataFrame([summary])


def build_tier_summary(user_detail: pd.DataFrame) -> pd.DataFrame:
    if user_detail.empty:
        return pd.DataFrame()

    tier_summary = (
        user_detail.groupby("governance_tier", as_index=False)
        .agg(
            users=("email", "count"),
            total_credits_used=("credits_used", "sum"),
            avg_credits_used=("credits_used", "mean"),
            median_credits_used=("credits_used", "median"),
            avg_cap_utilization=("cap_utilization", "mean"),
            users_over_80_percent_cap=("is_over_80_percent_cap", "sum"),
            users_over_90_percent_cap=("is_over_90_percent_cap", "sum"),
            users_at_or_over_cap=("is_at_or_over_cap", "sum"),
        )
    )

    total_credits = tier_summary["total_credits_used"].sum()
    tier_summary["share_of_total_credits"] = (
        tier_summary["total_credits_used"] / total_credits
        if total_credits > 0
        else 0
    )

    return tier_summary.sort_values("total_credits_used", ascending=False)


def print_summary(summary_df: pd.DataFrame, tier_summary: pd.DataFrame) -> None:
    row = summary_df.iloc[0]

    print("\nCap Pressure Summary")
    print("--------------------")
    print(f"Week:                         {row['week_start']} to {row['week_end']}")
    print(f"Users:                        {row['users']:,.0f}")
    print(f"Credit-active users:          {row['credit_active_users']:,.0f}")
    print(f"Total credits used:           {row['total_credits_used']:,.2f}")
    print(f"Average cap utilization:      {row['avg_cap_utilization']:.1%}")
    print(f"Median cap utilization:       {row['median_cap_utilization']:.1%}")

    print("\nThreshold Pressure")
    print("------------------")
    print(f"Users >= 80% cap:             {row['users_over_80_percent_cap']:,.0f}")
    print(f"Users >= 90% cap:             {row['users_over_90_percent_cap']:,.0f}")
    print(f"Users >= 100% cap:            {row['users_at_or_over_cap']:,.0f}")
    print(f"Users >= 110% cap:            {row['users_over_110_percent_cap']:,.0f}")

    if "emergency_override_users" in summary_df.columns:
        print(f"Emergency override users:     {row['emergency_override_users']:,.0f}")

    print("\nConcentration")
    print("-------------")
    print(f"Top 1% share:                 {row['top_1_percent_consumption_share']:.1%}")
    print(f"Top 5% share:                 {row['top_5_percent_consumption_share']:.1%}")
    print(f"Top 10% share:                {row['top_10_percent_consumption_share']:.1%}")
    print(f"Gini coefficient:             {row['gini_coefficient']:.3f}")
    print(f"HHI:                          {row['hhi']:.4f}")

    print("\nCap Pressure Index")
    print("------------------")
    print(f"CPI:                          {row['cap_pressure_index']:.1f}")
    print(f"Pressure level:               {row['pressure_level']}")

    if not tier_summary.empty:
        print("\nTier Summary")
        print("------------")
        for _, tier_row in tier_summary.iterrows():
            print(
                f"{tier_row['governance_tier']}: "
                f"{tier_row['users']:,.0f} users, "
                f"{tier_row['total_credits_used']:,.2f} credits, "
                f"{tier_row['avg_cap_utilization']:.1%} avg utilization"
            )

    print("\nFiles written")
    print("-------------")
    print(USER_DETAIL_OUTPUT)
    print(SUMMARY_OUTPUT)


def main() -> None:
    tier_config = load_tier_config()
    weekly_df = load_latest_week_usage()

    user_detail = build_user_detail(
        weekly_df=weekly_df,
        tier_config=tier_config,
    )

    summary = build_summary(user_detail)
    tier_summary = build_tier_summary(user_detail)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    user_detail.to_csv(USER_DETAIL_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    tier_summary_output = PROCESSED_DIR / "cap_pressure_tier_summary.csv"
    tier_summary.to_csv(tier_summary_output, index=False)

    print_summary(summary, tier_summary)
    print(tier_summary_output)


if __name__ == "__main__":
    main()
