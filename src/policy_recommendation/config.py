"""Configuration loading and validation for the policy recommendation package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import load_yaml


@dataclass(slots=True)
class PolicyRecommendationConfig:
    """Resolved configuration bundle for the recommendation engine."""

    project_root: Path
    config_dir: Path
    output_dir: Path
    legacy_processed_dir: Path
    recommendation: dict[str, Any]
    tier_policy: dict[str, Any]
    contract: dict[str, Any]
    policy_scenarios: dict[str, Any]

    @classmethod
    def load(cls, project_root: Path | None = None) -> "PolicyRecommendationConfig":
        project_root = project_root or Path(__file__).resolve().parents[2]
        config_dir = project_root / "config"
        output_dir = project_root / "outputs"
        legacy_processed_dir = project_root / "data" / "processed"

        recommendation = load_yaml(config_dir / "policy_recommendation_config.yaml")
        tier_policy = load_yaml(config_dir / "tier_policy_config.yaml")
        contract = load_yaml(config_dir / "contract_config.yaml")
        policy_scenarios = load_yaml(config_dir / "policy_scenarios.yaml")

        config = cls(
            project_root=project_root,
            config_dir=config_dir,
            output_dir=output_dir,
            legacy_processed_dir=legacy_processed_dir,
            recommendation=recommendation,
            tier_policy=tier_policy,
            contract=contract,
            policy_scenarios=policy_scenarios,
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate the recommendation config before any scoring logic runs."""

        required_sections = [
            "paths",
            "user_scoring",
            "user_action_thresholds",
            "tier_scoring",
            "tier_cap_thresholds",
            "tier_transition_rules",
            "contract_scoring",
            "contract_risk_thresholds",
            "confidence_rules",
            "reason_codes",
            "output_columns",
            "defaults",
        ]
        missing = [section for section in required_sections if section not in self.recommendation]
        if missing:
            raise ValueError(
                "policy_recommendation_config.yaml is missing required sections: "
                f"{missing}"
            )

        for section in ["user_scoring", "tier_scoring"]:
            score_section = self.recommendation.get(section, {})
            weights = score_section.get("weights")
            if not isinstance(weights, dict):
                raise ValueError(
                    f"policy_recommendation_config.yaml {section}.weights must be a mapping."
                )

        for section in ["reason_codes", "output_columns"]:
            if not isinstance(self.recommendation.get(section), dict):
                raise ValueError(
                    f"policy_recommendation_config.yaml {section} must be a mapping."
                )

    @property
    def reason_descriptions(self) -> dict[str, str]:
        return {
            str(key): str(value)
            for key, value in self.recommendation["reason_codes"].items()
        }

    @property
    def default_reason_summary(self) -> str:
        return str(
            self.recommendation.get("defaults", {}).get(
                "missing_reason_summary",
                "No material recommendation signals were triggered.",
            )
        )

    @property
    def complete_summary_note(self) -> str:
        return str(
            self.recommendation.get("defaults", {}).get(
                "summary_complete_note",
                "All recommendation sections were generated.",
            )
        )

    @property
    def warning_prefix(self) -> str:
        return str(self.recommendation.get("defaults", {}).get("warning_prefix", "Warning:"))

    @property
    def output_columns(self) -> dict[str, list[str]]:
        sections = self.recommendation.get("output_columns", {})
        return {
            str(section): list(columns)
            for section, columns in sections.items()
        }

    @property
    def input_roots(self) -> list[Path]:
        configured_roots = self.recommendation.get("paths", {}).get("input_roots", [])
        if configured_roots:
            return [self.project_root / Path(root) for root in configured_roots]
        return [self.output_dir, self.legacy_processed_dir]

    @property
    def output_root(self) -> Path:
        root = self.recommendation.get("paths", {}).get("output_root", "outputs")
        return self.project_root / Path(root)

    def get_section(self, section: str) -> dict[str, Any]:
        value = self.recommendation.get(section, {})
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(
                f"policy_recommendation_config.yaml section '{section}' must be a mapping."
            )
        return value

    def resolve_input_path(self, filename: str) -> Path | None:
        for root in self.input_roots:
            candidate = root / filename
            if candidate.exists():
                return candidate
        return None

    def load_optional_csv(self, filename: str) -> pd.DataFrame:
        path = self.resolve_input_path(filename)
        if path is None:
            print(f"{self.warning_prefix} missing optional input file '{filename}'.")
            return pd.DataFrame()
        df = pd.read_csv(path)
        print(f"Loaded {filename}: {path}")
        return df

    def load_single_row_optional_csv(self, filename: str) -> pd.Series | None:
        df = self.load_optional_csv(filename)
        if df.empty:
            return None
        return df.iloc[0]

    def build_tier_table(self) -> pd.DataFrame:
        tiers = self.tier_policy.get("tiers", {})
        if not tiers:
            raise ValueError("tier_policy_config.yaml must include a non-empty 'tiers' section.")

        transition_rules = self.get_section("tier_transition_rules")
        explicit_order = transition_rules.get("tier_order", [])

        if explicit_order:
            ordered_tiers = [str(tier_name) for tier_name in explicit_order if tier_name in tiers]
        else:
            ordered_tiers = [
                tier_name
                for tier_name, _ in sorted(
                    tiers.items(),
                    key=lambda item: float(item[1]["weekly_credit_cap"]),
                )
            ]

        rows = [
            {
                "tier": str(tier_name),
                "current_cap": float(tiers[tier_name]["weekly_credit_cap"]),
            }
            for tier_name in ordered_tiers
        ]

        tier_table = pd.DataFrame(rows).reset_index(drop=True)
        if tier_table.empty:
            raise ValueError("No tier cap rows could be built from tier_policy_config.yaml.")

        tier_table["tier_rank"] = range(1, len(tier_table) + 1)
        return tier_table
