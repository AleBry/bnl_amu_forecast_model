
from pathlib import Path
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONTRACT_CONFIG_FILE = PROJECT_ROOT / "config" / "contract_config.yaml"
TIER_POLICY_FILE = PROJECT_ROOT / "config" / "tier_policy_config.yaml"
SCENARIO_FILE = PROJECT_ROOT / "config" / "policy_scenarios.yaml"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FORECAST_FILE = PROCESSED_DIR / "forecast_summary.csv"
MONTE_CARLO_FILE = PROCESSED_DIR / "monte_carlo_summary.csv"
CONTRACT_STATUS_FILE = PROCESSED_DIR / "contract_status_summary.csv"
TIER_HISTORY_FILE = PROCESSED_DIR / "cap_pressure_history_tier_summary.csv"

SUMMARY_OUTPUT = PROCESSED_DIR / "policy_scenario_summary.csv"
COMPARISON_OUTPUT = PROCESSED_DIR / "policy_scenario_comparison.csv"


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_single_row_csv(path: Path, description: str) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{description} is empty: {path}")
    return df.iloc[0]


def load_tier_history() -> pd.DataFrame:
    if not TIER_HISTORY_FILE.exists():
        raise FileNotFoundError(
            f"Missing tier history file: {TIER_HISTORY_FILE}. "
            "Run cap_pressure_history.py first."
        )
    return pd.read_csv(TIER_HISTORY_FILE)


def build_tier_share_lookup(df: pd.DataFrame) -> dict:
    return {
        str(row["governance_tier"]): float(row["share_of_total_credits"])
        for _, row in df.iterrows()
    }


def build_tier_cap_lookup(config: dict) -> dict:
    return {
        tier: float(values["weekly_credit_cap"])
        for tier, values in config["tiers"].items()
    }


def calculate_dynamic_contract_size(
    scenario: dict,
    current_contract_size: float,
    forecast_balance: float,
) -> float:
    dynamic_cfg = scenario.get("dynamic_contract_size", {})

    method = dynamic_cfg.get("method")

    if method != "forecast_gap":
        raise ValueError(
            f"Unsupported dynamic_contract_size.method: {method}"
        )

    buffer_percent = float(dynamic_cfg.get("buffer_percent", 0))

    required_size = current_contract_size + abs(min(forecast_balance, 0))

    return required_size * (1 + buffer_percent / 100)


def resolve_purchased_credits(
    scenario: dict,
    current_contract_size: float,
    forecast_balance: float,
) -> float:

    if "dynamic_contract_size" in scenario:
        return calculate_dynamic_contract_size(
            scenario,
            current_contract_size,
            forecast_balance,
        )

    return float(
        scenario.get(
            "purchased_credits",
            current_contract_size,
        )
    )


def calculate_burn_adjustment_factor(
    scenario: dict,
    tier_shares: dict,
    tier_caps: dict,
    cap_change_sensitivity: float,
) -> float:

    total_effect = 0.0

    multipliers = scenario.get("tier_cap_multipliers", {})

    for tier, multiplier in multipliers.items():
        baseline_share = tier_shares.get(tier, 0.0)

        cap_change_pct = 1 - float(multiplier)

        total_effect += (
            baseline_share *
            cap_change_pct *
            cap_change_sensitivity
        )

    overrides = scenario.get("tier_cap_overrides", {})

    for tier, new_cap in overrides.items():
        baseline_cap = tier_caps.get(tier)

        if baseline_cap is None:
            continue

        multiplier = float(new_cap) / float(baseline_cap)

        cap_change_pct = 1 - multiplier

        total_effect += (
            tier_shares.get(tier, 0.0)
            * cap_change_pct
            * cap_change_sensitivity
        )

    return max(0.0, 1 - total_effect)


def estimate_risk(
    projected_balance: float,
    purchased_credits: float,
    baseline_balance: float,
    baseline_exhaustion_probability: float,
    assumptions: dict,
):

    exhaustion_probability = 0.0
    stranding_probability = 0.0

    if projected_balance < 0:

        denominator = abs(baseline_balance)

        if denominator <= 0:
            exhaustion_probability = baseline_exhaustion_probability

        else:

            severity_ratio = (
                abs(projected_balance)
                / denominator
            )

            exhaustion_probability = min(
                1.0,
                baseline_exhaustion_probability * severity_ratio,
            )

    elif projected_balance > 0:

        method = assumptions.get(
            "stranding_probability",
            {}
        ).get(
            "method",
            "surplus_ratio"
        )

        if method == "surplus_ratio":

            if purchased_credits > 0:

                stranding_probability = min(
                    1.0,
                    projected_balance / purchased_credits,
                )

        else:
            raise ValueError(
                f"Unsupported stranding probability method: {method}"
            )

    return exhaustion_probability, stranding_probability


def determine_scenario_status(
    exhaustion_probability: float,
    stranding_probability: float,
    thresholds: dict,
) -> str:

    if exhaustion_probability >= thresholds["high_exhaustion_probability"]:
        return "CRITICAL"

    if exhaustion_probability >= thresholds["moderate_exhaustion_probability"]:
        return "WARNING"

    if stranding_probability >= thresholds["high_stranding_probability"]:
        return "OVERSIZED"

    return "BALANCED"


def determine_balance_status(
    projected_balance: float,
    balance_config: dict,
) -> str:

    lower = float(balance_config["near_target_lower_bound"])
    upper = float(balance_config["near_target_upper_bound"])

    if projected_balance < lower:
        return "DEFICIT"

    if projected_balance > upper:
        return "SURPLUS"

    return "NEAR_TARGET"


def run_policy_scenarios():
    contract_config = load_yaml(CONTRACT_CONFIG_FILE)
    tier_config = load_yaml(TIER_POLICY_FILE)
    scenario_config = load_yaml(SCENARIO_FILE)

    forecast = load_single_row_csv(
        FORECAST_FILE,
        "forecast_summary.csv",
    )

    monte_carlo = load_single_row_csv(
        MONTE_CARLO_FILE,
        "monte_carlo_summary.csv",
    )

    contract_status = load_single_row_csv(
        CONTRACT_STATUS_FILE,
        "contract_status_summary.csv",
    )

    tier_history = load_tier_history()

    assumptions = scenario_config["assumptions"]

    cap_change_sensitivity = float(
        assumptions["cap_change_sensitivity"]
    )

    tier_shares = build_tier_share_lookup(tier_history)
    tier_caps = build_tier_cap_lookup(tier_config)

    current_contract_size = float(
        contract_status["purchased_credits"]
    )

    total_credits_used = float(
        contract_status["total_credits_used"]
    )

    weeks_remaining = float(
        contract_status["weeks_remaining"]
    )

    baseline_weekly_burn = float(
        forecast["forecast_weekly_burn"]
    )

    baseline_balance = float(
        monte_carlo["p50_contract_end_balance"]
    )

    baseline_exhaustion_probability = float(
        monte_carlo["exhaustion_probability"]
    )

    use_next_price = assumptions["cost_basis"]["use_next_contract_price"]

    price = (
        contract_config["pricing"]["next_contract_price_per_credit"]
        if use_next_price
        else contract_config["pricing"]["current_price_per_credit"]
    )

    rows = []

    for scenario in scenario_config["scenarios"]:

        scenario_name = scenario["name"]

        purchased_credits = resolve_purchased_credits(
            scenario,
            current_contract_size,
            float(forecast["forecast_contract_end_balance"]),
        )

        burn_adjustment_factor = calculate_burn_adjustment_factor(
            scenario,
            tier_shares,
            tier_caps,
            cap_change_sensitivity,
        )

        adjusted_weekly_burn = (
            baseline_weekly_burn *
            burn_adjustment_factor
        )

        future_usage = (
            adjusted_weekly_burn *
            weeks_remaining
        )

        projected_balance = (
            purchased_credits
            - total_credits_used
            - future_usage
        )

        exhaustion_probability, stranding_probability = estimate_risk(
            projected_balance,
            purchased_credits,
            baseline_balance,
            baseline_exhaustion_probability,
            assumptions,
        )

        scenario_status = determine_scenario_status(
            exhaustion_probability,
            stranding_probability,
            assumptions["risk_thresholds"],
        )

        balance_status = determine_balance_status(
            projected_balance,
            assumptions["balance_status"],
        )

        rows.append({
            "scenario_name": scenario_name,
            "purchased_credits": purchased_credits,
            "contract_size_delta": (
                purchased_credits - current_contract_size
            ),
            "burn_adjustment_factor": burn_adjustment_factor,
            "adjusted_weekly_burn": adjusted_weekly_burn,
            "projected_contract_end_balance": projected_balance,
            "absolute_balance_distance": abs(projected_balance),
            "estimated_credit_gap": max(0, -projected_balance),
            "estimated_credit_surplus": max(0, projected_balance),
            "estimated_exhaustion_probability": exhaustion_probability,
            "estimated_stranding_probability": stranding_probability,
            "scenario_status": scenario_status,
            "balance_status": balance_status,
            "estimated_contract_value": (
                purchased_credits * float(price)
            ),
        })
        
    summary_df = pd.DataFrame(rows)
    
    comparison_df = summary_df.copy()
    
    sort_columns = (
    scenario_config
        .get("comparison", {})
        .get(
            "sort_order",
        [
            "estimated_exhaustion_probability",
            "estimated_stranding_probability",
            "absolute_balance_distance",
        ],
    )
)
    comparison_df = comparison_df.sort_values(
        sort_columns
)
    return summary_df, comparison_df


def save_outputs(
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
):

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    comparison_df.to_csv(
        COMPARISON_OUTPUT,
        index=False,
    )


def print_summary(summary_df: pd.DataFrame):

    print("\\nPolicy Scenario Sandbox Complete")
    print("--------------------------------")
    print(f"Scenarios evaluated: {len(summary_df):,}")
    print("\\nFiles written")
    print("-------------")
    print(SUMMARY_OUTPUT)
    print(COMPARISON_OUTPUT)


def main():

    summary_df, comparison_df = run_policy_scenarios()

    save_outputs(
        summary_df,
        comparison_df,
    )

    print_summary(summary_df)


if __name__ == "__main__":
    main()
