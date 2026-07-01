"""Contract-level recommendation logic."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import PolicyRecommendationConfig
from .utils import build_reason_summary, get_nested_setting


class ContractPolicyRecommender:
    """Evaluate contract sizing and scenario-level recommendations."""

    def __init__(self, config: PolicyRecommendationConfig) -> None:
        self.config = config
        self.contract_thresholds = config.get_section("contract_risk_thresholds")
        self.contract_scoring = config.get_section("contract_scoring")
        self.reason_descriptions = config.reason_descriptions
        self.default_reason_summary = config.default_reason_summary
        self.output_columns = config.output_columns.get("contract", [])

    def recommend(
        self,
        forecast_row: pd.Series | None,
        monte_carlo_row: pd.Series | None,
        scenario_df: pd.DataFrame,
        contract_status_row: pd.Series | None,
    ) -> pd.DataFrame:
        """Return a one-row contract recommendation frame."""

        if forecast_row is None and monte_carlo_row is None and scenario_df.empty:
            return pd.DataFrame(
                columns=[
                    "current_contract_size",
                    "recommended_contract_size",
                    "recommended_delta",
                    "exhaustion_probability",
                    "stranding_probability",
                    "expected_end_balance",
                    "reason_summary",
                ]
            )

        best_scenario = self._select_best_contract_scenario(scenario_df)

        current_contract_size = None
        if contract_status_row is not None and "purchased_credits" in contract_status_row.index:
            current_contract_size = float(contract_status_row["purchased_credits"])
        elif monte_carlo_row is not None and "purchased_credits" in monte_carlo_row.index:
            current_contract_size = float(monte_carlo_row["purchased_credits"])

        recommended_contract_size = current_contract_size
        expected_end_balance = None
        exhaustion_probability = None
        stranding_probability = None
        reason_codes: list[str] = []

        if forecast_row is not None and "forecast_contract_end_balance" in forecast_row.index:
            expected_end_balance = float(forecast_row["forecast_contract_end_balance"])
            if expected_end_balance < 0:
                reason_codes.append("FORECAST_EXHAUSTION")

        if monte_carlo_row is not None:
            exhaustion_probability = float(monte_carlo_row.get("exhaustion_probability", 0) or 0)
            stranding_probability = float(monte_carlo_row.get("stranding_probability", 0) or 0)

            high_exhaustion = float(self.contract_thresholds.get("high_exhaustion_probability", 1) or 1)
            moderate_exhaustion = float(self.contract_thresholds.get("moderate_exhaustion_probability", 0) or 0)
            high_stranding = float(self.contract_thresholds.get("high_stranding_probability", 1) or 1)
            moderate_stranding = float(self.contract_thresholds.get("moderate_stranding_probability", 0) or 0)

            if exhaustion_probability >= moderate_exhaustion:
                reason_codes.append("MC_EXHAUSTION_HIGH")
            elif stranding_probability >= moderate_stranding:
                reason_codes.append("MC_STRANDING_HIGH")

            if exhaustion_probability >= high_exhaustion and "MC_EXHAUSTION_HIGH" not in reason_codes:
                reason_codes.append("MC_EXHAUSTION_HIGH")
            if stranding_probability >= high_stranding and "MC_STRANDING_HIGH" not in reason_codes:
                reason_codes.append("MC_STRANDING_HIGH")

        if best_scenario is not None:
            if bool(self.contract_scoring.get("use_best_scenario_sort", True)):
                recommended_contract_size = float(best_scenario.get("purchased_credits", recommended_contract_size))
            if str(best_scenario.get("scenario_name", "")) == str(
                self.contract_scoring.get("preferred_scenario_name", "Current Policy")
            ):
                reason_codes.append("SCENARIO_CURRENT_POLICY")
            else:
                reason_codes.append("SCENARIO_BETTER_OPTION")

        reason_codes = list(dict.fromkeys(reason_codes))
        recommended_delta = None
        if current_contract_size is not None and recommended_contract_size is not None:
            recommended_delta = recommended_contract_size - current_contract_size

        result = pd.DataFrame(
            [
                {
                    "current_contract_size": current_contract_size,
                    "recommended_contract_size": recommended_contract_size,
                    "recommended_delta": recommended_delta,
                    "exhaustion_probability": exhaustion_probability,
                    "stranding_probability": stranding_probability,
                    "expected_end_balance": expected_end_balance,
                    "reason_summary": build_reason_summary(
                        reason_codes,
                        self.reason_descriptions,
                        self.default_reason_summary,
                    ),
                }
            ]
        )

        return result[[column for column in self.output_columns if column in result.columns]]

    def _select_best_contract_scenario(self, scenario_df: pd.DataFrame) -> pd.Series | None:
        if scenario_df.empty:
            return None

        sort_columns = (
            self.config.policy_scenarios.get("comparison", {}).get(
                "sort_order",
                [],
            )
        )
        available_sort_columns = [column for column in sort_columns if column in scenario_df.columns]
        if not available_sort_columns:
            return scenario_df.iloc[0]

        return scenario_df.sort_values(available_sort_columns, ascending=True).iloc[0]
