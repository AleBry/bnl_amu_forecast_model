"""User-level recommendation logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import PolicyRecommendationConfig
from .utils import (
    build_reason_summary,
    calculate_confidence,
    coerce_numeric_columns,
    get_nested_setting,
    normalize_bool,
)


@dataclass(slots=True)
class UserRecommendationResult:
    """User recommendation outputs and credit-impact summary."""

    user_recommendations: pd.DataFrame
    credit_impact_summary: pd.DataFrame


class UserPolicyRecommender:
    """Score users and map them to adjacent governance tiers."""

    def __init__(self, config: PolicyRecommendationConfig, tier_table: pd.DataFrame) -> None:
        self.config = config
        self.tier_table = tier_table.reset_index(drop=True)
        self.transition_rules = config.get_section("tier_transition_rules")
        self.user_thresholds = config.get_section("user_action_thresholds")
        self.user_scoring = config.get_section("user_scoring")
        self.confidence_rules = config.get_section("confidence_rules")
        self.reason_descriptions = config.reason_descriptions
        self.default_reason_summary = config.default_reason_summary
        self.user_output_columns = config.output_columns.get("user", [])
        self.credit_summary_columns = config.output_columns.get("credit_impact_summary", [])
        self.weights = self.user_scoring.get("weights", {})
        self.max_score = float(self.user_scoring.get("score_maximum", 4.0))

        if not isinstance(self.weights, dict):
            raise ValueError("policy_recommendation_config.yaml user_scoring.weights must be a mapping.")

        self.tier_rank_lookup = {
            row["tier"]: int(row["tier_rank"])
            for _, row in self.tier_table.iterrows()
        }
        self.tier_cap_lookup = {
            row["tier"]: float(row["current_cap"])
            for _, row in self.tier_table.iterrows()
        }

    def recommend(
        self,
        user_pressure_df: pd.DataFrame,
        segment_df: pd.DataFrame,
    ) -> UserRecommendationResult:
        """Return user recommendations and the total tier-change impact summary."""

        if user_pressure_df.empty:
            empty = pd.DataFrame(
                columns=[
                    "user_id",
                    "current_tier",
                    "recommended_action",
                    "recommended_tier",
                    "tier_change_direction",
                    "tier_change_reason",
                    "confidence",
                    "reason_codes",
                    "reason_summary",
                    "estimated_credit_impact",
                    "net_credit_impact",
                ]
            )
            summary = pd.DataFrame(
                [
                    {
                        "current_total_estimated_credit_impact": 0.0,
                        "recommended_total_estimated_credit_impact": 0.0,
                        "net_credit_impact": 0.0,
                        "num_users_recommended_up": 0,
                        "num_users_recommended_down": 0,
                        "num_users_reviewed": 0,
                        "notes": "No user recommendations were generated.",
                    }
                ]
            )
            return UserRecommendationResult(empty, summary)

        df = coerce_numeric_columns(
            user_pressure_df,
            [
                "weeks_observed",
                "avg_weekly_credits_used",
                "latest_weekly_credit_cap",
                "avg_cap_utilization",
                "latest_cap_utilization",
                "share_weeks_over_90_percent_cap",
            ],
        ).copy()

        if not segment_df.empty and "email" in segment_df.columns:
            segment_lookup = segment_df[["email", "usage_segment"]].drop_duplicates()
            df = df.merge(segment_lookup, how="left", on="email")

        segment_rank_lookup = self._build_user_segment_rank_lookup(segment_df)
        rows: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            user_row = self._recommend_one(row, segment_rank_lookup)
            rows.append(user_row)

        result = pd.DataFrame(rows)
        result = self._apply_output_order(result)

        credit_summary = self._build_credit_impact_summary(result)
        result = result.drop(columns=["current_credit_cap", "recommended_credit_cap"], errors="ignore")
        result = result[[column for column in self.user_output_columns if column in result.columns]]

        return UserRecommendationResult(result, credit_summary)

    def _apply_output_order(self, result: pd.DataFrame) -> pd.DataFrame:
        if result.empty:
            return result

        action_order = {"REVIEW": 1, "MOVE_UP": 2, "MOVE_DOWN": 3, "MAINTAIN": 4}
        confidence_order = {"HIGH": 1, "MEDIUM": 2, "LOW": 3}

        result["_action_sort"] = result["recommended_action"].map(action_order).fillna(99)
        result["_confidence_sort"] = result["confidence"].map(confidence_order).fillna(99)
        result = result.sort_values(
            ["_action_sort", "_confidence_sort", "current_tier", "user_id"],
            ascending=[True, True, True, True],
        ).drop(columns=["_action_sort", "_confidence_sort"])
        return result

    def _recommend_one(
        self,
        row: pd.Series,
        segment_rank_lookup: dict[str, int],
    ) -> dict[str, Any]:
        current_tier = str(row.get("latest_governance_tier", "UNKNOWN"))
        weeks_observed = int(pd.to_numeric(row.get("weeks_observed"), errors="coerce") or 0)
        avg_utilization = float(pd.to_numeric(row.get("avg_cap_utilization"), errors="coerce") or 0)
        latest_utilization = float(pd.to_numeric(row.get("latest_cap_utilization"), errors="coerce") or 0)
        share_over_90 = float(pd.to_numeric(row.get("share_weeks_over_90_percent_cap"), errors="coerce") or 0)
        current_cap = pd.to_numeric(row.get("latest_weekly_credit_cap"), errors="coerce")
        usage_segment = row.get("usage_segment")

        move_up_min_weeks = int(get_nested_setting(self.user_thresholds, ["move_up", "minimum_weeks_observed"], 0) or 0)
        move_up_share_threshold = float(get_nested_setting(self.user_thresholds, ["move_up", "minimum_share_over_90_pct"], 0) or 0)
        move_down_min_weeks = int(get_nested_setting(self.user_thresholds, ["move_down", "minimum_weeks_observed"], 0) or 0)
        move_down_avg_threshold = float(get_nested_setting(self.user_thresholds, ["move_down", "maximum_average_utilization"], 0) or 0)
        spike_threshold = float(get_nested_setting(self.user_thresholds, ["review", "spike_utilization_threshold"], 0) or 0)

        move_up_signal = weeks_observed >= move_up_min_weeks and share_over_90 >= move_up_share_threshold
        move_down_signal = weeks_observed >= move_down_min_weeks and avg_utilization <= move_down_avg_threshold
        spike_signal = latest_utilization >= spike_threshold and not move_up_signal
        emergency_signal = normalize_bool(row.get("ever_emergency_override"))
        limited_history_signal = (
            weeks_observed < min(value for value in [move_up_min_weeks, move_down_min_weeks] if value > 0)
            if any(value > 0 for value in [move_up_min_weeks, move_down_min_weeks])
            else False
        )

        segment_rank = segment_rank_lookup.get(str(usage_segment), self.tier_rank_lookup.get(current_tier, 0))
        current_rank = self.tier_rank_lookup.get(current_tier, 0)
        historical_heavy_signal = segment_rank > current_rank
        historical_light_signal = segment_rank < current_rank

        move_up_score = 0.0
        move_down_score = 0.0
        review_score = 0.0
        reason_codes: list[str] = []

        move_up_pressure_weight = float(self.weights.get("move_up_pressure", 0))
        historical_heavy_weight = float(self.weights.get("historical_heavy_user", 0))
        move_down_pressure_weight = float(self.weights.get("move_down_pressure", 0))
        historical_light_weight = float(self.weights.get("historical_light_user", 0))
        review_emergency_weight = float(self.weights.get("review_emergency_override", 0))
        review_history_weight = float(self.weights.get("review_limited_history", 0))
        review_spike_weight = float(self.weights.get("review_recent_spike", 0))

        if move_up_signal:
            move_up_score += move_up_pressure_weight
            reason_codes.append("PRESSURE_HIGH")
        if historical_heavy_signal:
            move_up_score += historical_heavy_weight
            reason_codes.append("HISTORICAL_HEAVY_USER")
        if move_down_signal:
            move_down_score += move_down_pressure_weight
            reason_codes.append("PRESSURE_LOW")
        if historical_light_signal:
            move_down_score += historical_light_weight
            reason_codes.append("HISTORICAL_LIGHT_USER")
        if emergency_signal:
            review_score += review_emergency_weight
            reason_codes.append("EMERGENCY_OVERRIDE")
        if limited_history_signal:
            review_score += review_history_weight
            reason_codes.append("HISTORY_LIMITED")
        if spike_signal:
            review_score += review_spike_weight
            reason_codes.append("PRESSURE_SPIKE")

        if review_score >= max(move_up_score, move_down_score) and review_score > 0:
            recommended_action = "REVIEW"
            recommendation_score = review_score
        elif move_up_score > move_down_score:
            recommended_action = "MOVE_UP"
            recommendation_score = move_up_score
        elif move_down_score > move_up_score:
            recommended_action = "MOVE_DOWN"
            recommendation_score = move_down_score
        else:
            recommended_action = "MAINTAIN"
            recommendation_score = max(move_up_score, move_down_score, review_score, 0.0)

        recommended_tier, boundary_reason_codes, resolved_action = self._transition_user_tier(
            current_tier=current_tier,
            recommended_action=recommended_action,
        )
        if boundary_reason_codes:
            reason_codes.extend(boundary_reason_codes)
        if resolved_action == "REVIEW" and recommended_action in {"MOVE_UP", "MOVE_DOWN"}:
            recommended_action = "REVIEW"

        current_credit_cap = self._get_tier_cap(current_tier)
        recommended_credit_cap = self._get_tier_cap(recommended_tier)
        estimated_credit_impact = (
            float(recommended_credit_cap - current_credit_cap)
            if current_credit_cap is not None and recommended_credit_cap is not None
            else None
        )

        change_direction = "NONE"
        if recommended_tier != current_tier:
            if recommended_credit_cap is not None and current_credit_cap is not None:
                change_direction = "UP" if recommended_credit_cap > current_credit_cap else "DOWN"

        reason_codes = list(dict.fromkeys(reason_codes))

        return {
            "user_id": row.get("email"),
            "current_tier": current_tier,
            "recommended_action": recommended_action,
            "recommended_tier": recommended_tier,
            "tier_change_direction": change_direction,
            "tier_change_reason": "|".join(boundary_reason_codes),
            "confidence": calculate_confidence(recommendation_score, self.max_score, self.confidence_rules),
            "reason_codes": "|".join(reason_codes),
            "reason_summary": build_reason_summary(
                reason_codes,
                self.reason_descriptions,
                self.default_reason_summary,
            ),
            "estimated_credit_impact": estimated_credit_impact,
            "net_credit_impact": estimated_credit_impact,
            "current_credit_cap": current_credit_cap,
            "recommended_credit_cap": recommended_credit_cap,
        }

    def _transition_user_tier(
        self,
        current_tier: str,
        recommended_action: str,
    ) -> tuple[str, list[str], str]:
        tiers = list(self.tier_table["tier"])
        lowest_tier = tiers[0] if tiers else current_tier
        highest_tier = tiers[-1] if tiers else current_tier
        lower_tier, upper_tier = self._get_tier_neighbors(current_tier)
        boundary_codes = self.transition_rules.get("boundary_reason_codes", {}) or {}
        fallback_action = str(self.transition_rules.get("boundary_fallback_action", "REVIEW"))
        review_behavior = str(self.transition_rules.get("review_tier_behavior", "preserve_current"))
        reason_codes: list[str] = []

        if recommended_action == "MOVE_UP":
            if current_tier == highest_tier or upper_tier == current_tier:
                reason_codes.append(str(boundary_codes.get("no_higher_tier", "TIER_BOUNDARY_HIGH")))
                return current_tier, reason_codes, fallback_action
            return upper_tier, reason_codes, recommended_action

        if recommended_action == "MOVE_DOWN":
            if current_tier == lowest_tier or lower_tier == current_tier:
                reason_codes.append(str(boundary_codes.get("no_lower_tier", "TIER_BOUNDARY_LOW")))
                return current_tier, reason_codes, fallback_action
            return lower_tier, reason_codes, recommended_action

        if recommended_action == "REVIEW" and review_behavior == "special_placeholder":
            return "REVIEW", reason_codes, recommended_action

        return current_tier, reason_codes, recommended_action

    def _get_tier_neighbors(self, current_tier: str) -> tuple[str, str]:
        tiers = list(self.tier_table["tier"])
        if current_tier not in tiers:
            return current_tier, current_tier
        current_index = tiers.index(current_tier)
        lower_tier = tiers[max(current_index - 1, 0)]
        upper_tier = tiers[min(current_index + 1, len(tiers) - 1)]
        return lower_tier, upper_tier

    def _get_tier_cap(self, tier_name: str) -> float | None:
        return self.tier_cap_lookup.get(tier_name)

    def _build_user_segment_rank_lookup(self, segment_df: pd.DataFrame) -> dict[str, int]:
        if segment_df.empty or "usage_segment" not in segment_df.columns:
            return {}

        tier_names = list(self.tier_table["tier"])
        rank_lookup = {tier: rank for rank, tier in enumerate(tier_names, start=1)}
        segment_rank_lookup: dict[str, int] = {}

        for segment in segment_df["usage_segment"].dropna().unique():
            segment_text = str(segment)
            assigned_rank = None

            for tier_name in reversed(tier_names):
                if tier_name.lower() in segment_text.lower():
                    assigned_rank = rank_lookup[tier_name]
                    break

            if assigned_rank is None and "above" in segment_text.lower():
                assigned_rank = len(tier_names)

            if assigned_rank is not None:
                segment_rank_lookup[segment_text] = assigned_rank

        return segment_rank_lookup

    def _build_credit_impact_summary(self, result: pd.DataFrame) -> pd.DataFrame:
        changed = result[result["recommended_tier"] != result["current_tier"]].copy()
        if changed.empty:
            return pd.DataFrame(
                [
                    {
                        "current_total_estimated_credit_impact": 0.0,
                        "recommended_total_estimated_credit_impact": 0.0,
                        "net_credit_impact": 0.0,
                        "num_users_recommended_up": 0,
                        "num_users_recommended_down": 0,
                        "num_users_reviewed": int((result["recommended_action"] == "REVIEW").sum()),
                        "notes": "No users were recommended to change tier.",
                    }
                ]
            )

        current_total = float(changed["current_credit_cap"].fillna(0).sum())
        recommended_total = float(changed["recommended_credit_cap"].fillna(0).sum())
        net_credit_impact = recommended_total - current_total

        summary = pd.DataFrame(
            [
                {
                    "current_total_estimated_credit_impact": current_total,
                    "recommended_total_estimated_credit_impact": recommended_total,
                    "net_credit_impact": net_credit_impact,
                    "num_users_recommended_up": int((changed["tier_change_direction"] == "UP").sum()),
                    "num_users_recommended_down": int((changed["tier_change_direction"] == "DOWN").sum()),
                    "num_users_reviewed": int((result["recommended_action"] == "REVIEW").sum()),
                    "notes": (
                        "Net credit impact is calculated as recommended minus current tier cap "
                        "for all users whose recommended tier differs from their current tier."
                    ),
                }
            ]
        )

        return summary[[column for column in self.credit_summary_columns if column in summary.columns]]
