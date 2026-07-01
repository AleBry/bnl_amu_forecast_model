"""Lightweight data structures for the policy recommendation package."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class RecommendationInputs:
    """Container for the input frames loaded by the orchestration engine."""

    historical_user_segments: pd.DataFrame
    cap_pressure_history_user_summary: pd.DataFrame
    cap_pressure_history_tier_summary: pd.DataFrame
    tier_recommendations: pd.DataFrame
    forecast_summary: pd.Series | None
    monte_carlo_summary: pd.Series | None
    policy_scenario_summary: pd.DataFrame
    contract_status_summary: pd.Series | None


@dataclass(slots=True)
class RecommendationOutputs:
    """Container for the output frames produced by the orchestration engine."""

    user_recommendations: pd.DataFrame
    tier_recommendations: pd.DataFrame
    contract_recommendations: pd.DataFrame
    summary: pd.DataFrame
    credit_impact_summary: pd.DataFrame
