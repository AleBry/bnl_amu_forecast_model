"""Thin CLI entry point for the policy recommendation package."""

from __future__ import annotations

from policy_recommendation.engine import PolicyRecommendationEngine, main

__all__ = ["PolicyRecommendationEngine", "main"]


if __name__ == "__main__":
    main()
