"""Top-level orchestration for the policy recommendation package."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .config import PolicyRecommendationConfig
from .contracts import ContractPolicyRecommender
from .models import RecommendationInputs, RecommendationOutputs
from .tiers import TierPolicyRecommender
from .users import UserPolicyRecommender
from .utils import select_output_columns


class PolicyRecommendationEngine:
    """Load inputs, run the recommenders, and write all outputs."""

    def __init__(self, config: PolicyRecommendationConfig | None = None) -> None:
        self.config = config or PolicyRecommendationConfig.load()
        self.tier_table = self.config.build_tier_table()
        self.user_recommender = UserPolicyRecommender(self.config, self.tier_table)
        self.tier_recommender = TierPolicyRecommender(self.config, self.tier_table)
        self.contract_recommender = ContractPolicyRecommender(self.config)

        self.output_dir = self.config.output_root
        self.user_output_file = self.output_dir / "policy_recommendation_user_recommendations.csv"
        self.tier_output_file = self.output_dir / "policy_recommendation_tier_recommendations.csv"
        self.contract_output_file = self.output_dir / "policy_recommendation_contract_recommendations.csv"
        self.summary_output_file = self.output_dir / "policy_recommendation_summary.csv"
        self.credit_impact_output_file = self.output_dir / "policy_recommendation_credit_impact_summary.csv"

    def load_inputs(self) -> RecommendationInputs:
        """Load all current pipeline inputs with graceful optional-file handling."""

        inputs = RecommendationInputs(
            historical_user_segments=self.config.load_optional_csv("historical_user_segments.csv"),
            cap_pressure_history_user_summary=self.config.load_optional_csv("cap_pressure_history_user_summary.csv"),
            cap_pressure_history_tier_summary=self.config.load_optional_csv("cap_pressure_history_tier_summary.csv"),
            tier_recommendations=self.config.load_optional_csv("tier_recommendations.csv"),
            forecast_summary=self.config.load_single_row_optional_csv("forecast_summary.csv"),
            monte_carlo_summary=self.config.load_single_row_optional_csv("monte_carlo_summary.csv"),
            policy_scenario_summary=self.config.load_optional_csv("policy_scenario_summary.csv"),
            contract_status_summary=self.config.load_single_row_optional_csv("contract_status_summary.csv"),
        )

        self._validate_essential_inputs(inputs)
        return inputs

    def run(self) -> RecommendationOutputs:
        """Run the recommendation workflow end to end."""

        print("Starting policy recommendation engine...")
        inputs = self.load_inputs()

        user_result = self.user_recommender.recommend(
            user_pressure_df=inputs.cap_pressure_history_user_summary,
            segment_df=inputs.historical_user_segments,
        )
        tier_recommendations = self.tier_recommender.recommend(
            tier_pressure_df=inputs.cap_pressure_history_tier_summary,
        )
        contract_recommendations = self.contract_recommender.recommend(
            forecast_row=inputs.forecast_summary,
            monte_carlo_row=inputs.monte_carlo_summary,
            scenario_df=inputs.policy_scenario_summary,
            contract_status_row=inputs.contract_status_summary,
        )

        summary = self._build_summary(
            user_recommendations=user_result.user_recommendations,
            tier_recommendations=tier_recommendations,
            contract_recommendations=contract_recommendations,
            credit_impact_summary=user_result.credit_impact_summary,
        )
        credit_impact_summary = user_result.credit_impact_summary

        self._print_user_diagnostics(user_result.user_recommendations)

        outputs = RecommendationOutputs(
            user_recommendations=user_result.user_recommendations,
            tier_recommendations=tier_recommendations,
            contract_recommendations=contract_recommendations,
            summary=summary,
            credit_impact_summary=credit_impact_summary,
        )
        self.write_outputs(outputs)

        print("Policy recommendation engine complete.")
        return outputs

    def write_outputs(self, outputs: RecommendationOutputs) -> None:
        """Write all generated outputs to the configured output directory."""

        self._write_output(outputs.user_recommendations, self.user_output_file)
        self._write_output(outputs.tier_recommendations, self.tier_output_file)
        self._write_output(outputs.contract_recommendations, self.contract_output_file)
        self._write_output(outputs.summary, self.summary_output_file)
        self._write_output(outputs.credit_impact_summary, self.credit_impact_output_file)

    def _validate_essential_inputs(self, inputs: RecommendationInputs) -> None:
        if (
            inputs.cap_pressure_history_user_summary.empty
            and inputs.forecast_summary is None
            and inputs.monte_carlo_summary is None
            and inputs.policy_scenario_summary.empty
        ):
            raise FileNotFoundError(
                "No essential recommendation inputs were found. Expected at least one of: "
                "cap_pressure_history_user_summary.csv, forecast_summary.csv, "
                "monte_carlo_summary.csv, or policy_scenario_summary.csv in outputs/ "
                "or data/processed/."
            )

    def _build_summary(
        self,
        user_recommendations: pd.DataFrame,
        tier_recommendations: pd.DataFrame,
        contract_recommendations: pd.DataFrame,
        credit_impact_summary: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build the top-level run summary."""

        run_timestamp = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        contract_recommendation = ""
        if not contract_recommendations.empty:
            contract_row = contract_recommendations.iloc[0]
            current_size = contract_row.get("current_contract_size")
            recommended_size = contract_row.get("recommended_contract_size")
            if pd.notna(current_size) and pd.notna(recommended_size):
                contract_recommendation = (
                    f"Recommend contract size {recommended_size:,.0f} from current {current_size:,.0f}."
                )

        primary_risks: list[str] = []
        if not user_recommendations.empty:
            if (user_recommendations["recommended_action"] == "REVIEW").any():
                primary_risks.append("user_review_backlog")
            if (user_recommendations["recommended_action"] == "MOVE_UP").any():
                primary_risks.append("user_capacity_pressure")
            if (user_recommendations["recommended_action"] == "MOVE_DOWN").any():
                primary_risks.append("possible_over-allocation")

        if not tier_recommendations.empty:
            if (tier_recommendations["cap_direction"] == "TOO_LOW").any():
                primary_risks.append("tier_caps_too_low")
            if (tier_recommendations["cap_direction"] == "TOO_HIGH").any():
                primary_risks.append("tier_caps_too_high")

        net_credit_impact = 0.0
        if not credit_impact_summary.empty and "net_credit_impact" in credit_impact_summary.columns:
            net_credit_impact = float(credit_impact_summary.iloc[0]["net_credit_impact"] or 0)

        notes_parts: list[str] = []
        if user_recommendations.empty:
            notes_parts.append("User-level recommendation section unavailable.")
        if tier_recommendations.empty:
            notes_parts.append("Tier-level recommendation section unavailable.")
        if contract_recommendations.empty:
            notes_parts.append("Contract-level recommendation section unavailable.")

        summary = pd.DataFrame(
            [
                {
                    "run_timestamp": run_timestamp,
                    "num_users_reviewed": int(len(user_recommendations)),
                    "num_user_moves_up": int((user_recommendations.get("recommended_action") == "MOVE_UP").sum()) if not user_recommendations.empty else 0,
                    "num_user_moves_down": int((user_recommendations.get("recommended_action") == "MOVE_DOWN").sum()) if not user_recommendations.empty else 0,
                    "num_users_retain": int((user_recommendations.get("recommended_action") == "MAINTAIN").sum()) if not user_recommendations.empty else 0,
                    "num_users_review": int((user_recommendations.get("recommended_action") == "REVIEW").sum()) if not user_recommendations.empty else 0,
                    "num_tiers_adjusted": int((tier_recommendations.get("cap_direction").isin(["TOO_LOW", "TOO_HIGH"])).sum()) if not tier_recommendations.empty else 0,
                    "contract_recommendation": contract_recommendation,
                    "net_credit_impact": net_credit_impact,
                    "primary_risks": "|".join(dict.fromkeys(primary_risks)),
                    "notes": " ".join(notes_parts) if notes_parts else self.config.complete_summary_note,
                }
            ]
        )

        return select_output_columns(summary, self.config.output_columns.get("summary", []))

    def _print_user_diagnostics(self, user_recommendations: pd.DataFrame) -> None:
        """Print a simple health check for user-level movement."""

        if user_recommendations.empty:
            print("Warning: no user recommendations were generated.")
            return

        counts = user_recommendations["recommended_action"].value_counts().to_dict()
        changed_count = int((user_recommendations["recommended_tier"] != user_recommendations["current_tier"]).sum())

        print("\nUser Recommendation Diagnostics")
        print("-------------------------------")
        for action in ["MOVE_UP", "MOVE_DOWN", "MAINTAIN", "REVIEW"]:
            print(f"{action}: {int(counts.get(action, 0))}")
        print(f"Users with tier movement: {changed_count}")

        if changed_count == 0:
            print("Warning: all recommended tiers match the current tier.")
        elif changed_count > 0:
            sample = user_recommendations.loc[
                user_recommendations["recommended_tier"] != user_recommendations["current_tier"],
                [
                    "user_id",
                    "current_tier",
                    "recommended_action",
                    "recommended_tier",
                    "tier_change_direction",
                    "tier_change_reason",
                ],
            ].head(5)
            print("\nSample Tier Movements")
            print("---------------------")
            for _, row in sample.iterrows():
                print(
                    f"{row['user_id']}: {row['current_tier']} -> {row['recommended_tier']} "
                    f"({row['recommended_action']}, {row['tier_change_direction']})"
                )

    def _write_output(self, df: pd.DataFrame, path: Path) -> None:
        """Write a CSV and fall back to a pending file if the target is locked."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            df.to_csv(path, index=False)
            print(f"Wrote output: {path}")
        except PermissionError as exc:
            fallback_path = path.with_name(f"{path.stem}.pending{path.suffix}")
            df.to_csv(fallback_path, index=False)
            print(
                f"Warning: could not overwrite {path} because it is locked. "
                f"Wrote fallback output instead: {fallback_path}. Details: {exc}"
            )


def main() -> None:
    """Run the policy recommendation engine from the project root."""

    engine = PolicyRecommendationEngine()
    engine.run()
