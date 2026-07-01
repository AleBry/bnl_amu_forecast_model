"""Tier-level recommendation logic."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import PolicyRecommendationConfig
from .utils import build_reason_summary, calculate_confidence, coerce_numeric_columns, get_nested_setting


class TierPolicyRecommender:
    """Evaluate whether current tier caps appear too low, too high, or aligned."""

    def __init__(self, config: PolicyRecommendationConfig, tier_table: pd.DataFrame) -> None:
        self.config = config
        self.tier_table = tier_table.reset_index(drop=True)
        self.tier_thresholds = config.get_section("tier_cap_thresholds")
        self.tier_scoring = config.get_section("tier_scoring")
        self.confidence_rules = config.get_section("confidence_rules")
        self.reason_descriptions = config.reason_descriptions
        self.default_reason_summary = config.default_reason_summary
        self.output_columns = config.output_columns.get("tier", [])
        self.max_score = float(self.tier_scoring.get("score_maximum", 2.0))

    def recommend(self, tier_pressure_df: pd.DataFrame) -> pd.DataFrame:
        """Return tier cap alignment recommendations."""

        if tier_pressure_df.empty:
            return pd.DataFrame(
                columns=[
                    "tier",
                    "current_cap",
                    "suggested_cap",
                    "cap_direction",
                    "confidence",
                    "reason_codes",
                    "reason_summary",
                ]
            )

        pressure_df = coerce_numeric_columns(
            tier_pressure_df,
            ["avg_cap_utilization", "share_user_weeks_over_90_percent_cap"],
        ).copy()

        pressure_df = pressure_df.rename(columns={"governance_tier": "tier"})
        pressure_df = pressure_df.merge(
            self.tier_table[["tier", "current_cap", "tier_rank"]],
            how="left",
            on="tier",
        )

        move_up_share_threshold = float(self.tier_thresholds.get("overutilization_share_for_raise", 0) or 0)
        move_down_avg_threshold = float(self.tier_thresholds.get("average_utilization_for_lowering", 0) or 0)
        too_low_weight = float(get_nested_setting(self.tier_scoring, ["weights", "too_low_pressure"], 0) or 0)
        too_high_weight = float(get_nested_setting(self.tier_scoring, ["weights", "too_high_pressure"], 0) or 0)

        rows: list[dict[str, Any]] = []
        for _, row in pressure_df.iterrows():
            tier = str(row["tier"])
            current_cap = float(row["current_cap"])
            avg_utilization = float(row.get("avg_cap_utilization", 0) or 0)
            share_over_90 = float(row.get("share_user_weeks_over_90_percent_cap", 0) or 0)

            lower_tier, upper_tier = self._get_tier_neighbors(tier)
            lower_cap = self._get_tier_cap(lower_tier) or current_cap
            upper_cap = self._get_tier_cap(upper_tier) or current_cap

            increase_score = 0.0
            decrease_score = 0.0
            reason_codes: list[str] = []

            if share_over_90 >= move_up_share_threshold and upper_cap > current_cap:
                increase_score += too_low_weight
                reason_codes.append("TIER_PRESSURE_HIGH")

            if avg_utilization <= move_down_avg_threshold and lower_cap < current_cap:
                decrease_score += too_high_weight
                reason_codes.append("TIER_PRESSURE_LOW")

            if increase_score > decrease_score:
                suggested_cap = upper_cap
                cap_direction = "TOO_LOW"
                score = increase_score
            elif decrease_score > increase_score:
                suggested_cap = lower_cap
                cap_direction = "TOO_HIGH"
                score = decrease_score
            else:
                suggested_cap = current_cap
                cap_direction = "ALIGNED"
                score = 0.0
                reason_codes.append("CAP_ALIGNED")

            reason_codes = list(dict.fromkeys(reason_codes))
            rows.append(
                {
                    "tier": tier,
                    "current_cap": current_cap,
                    "suggested_cap": suggested_cap,
                    "cap_direction": cap_direction,
                    "confidence": calculate_confidence(score, self.max_score, self.confidence_rules),
                    "reason_codes": "|".join(reason_codes),
                    "reason_summary": build_reason_summary(
                        reason_codes,
                        self.reason_descriptions,
                        self.default_reason_summary,
                    ),
                }
            )

        result = pd.DataFrame(rows)
        if result.empty:
            return result

        direction_order = {"TOO_LOW": 1, "TOO_HIGH": 2, "ALIGNED": 3}
        result["_direction_sort"] = result["cap_direction"].map(direction_order).fillna(99)
        result = result.sort_values(["_direction_sort", "current_cap", "tier"]).drop(columns="_direction_sort")
        return result[[column for column in self.output_columns if column in result.columns]]

    def _get_tier_neighbors(self, current_tier: str) -> tuple[str, str]:
        tiers = list(self.tier_table["tier"])
        if current_tier not in tiers:
            return current_tier, current_tier
        current_index = tiers.index(current_tier)
        lower_tier = tiers[max(current_index - 1, 0)]
        upper_tier = tiers[min(current_index + 1, len(tiers) - 1)]
        return lower_tier, upper_tier

    def _get_tier_cap(self, tier_name: str) -> float | None:
        matches = self.tier_table.loc[self.tier_table["tier"] == tier_name, "current_cap"]
        if matches.empty:
            return None
        return float(matches.iloc[0])
