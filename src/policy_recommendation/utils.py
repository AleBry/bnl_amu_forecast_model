"""Shared helpers for the policy recommendation package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return an empty mapping when the file is blank."""

    if not path.exists():
        raise FileNotFoundError(f"Missing required config file: {path}")

    with open(path, "r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj)

    return data or {}


def coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert selected columns to numeric values without raising on bad data."""

    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def normalize_bool(value: Any) -> bool:
    """Interpret common string and numeric representations of truthy values."""

    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def get_nested_setting(config: dict[str, Any], path: list[str], default: Any = None) -> Any:
    """Retrieve a nested mapping value with a fallback."""

    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def build_reason_summary(
    reason_codes: list[str],
    reason_descriptions: dict[str, str],
    default_message: str,
) -> str:
    """Expand reason codes into a readable explanation string."""

    if not reason_codes:
        return default_message
    return " ".join(
        reason_descriptions[code]
        for code in reason_codes
        if code in reason_descriptions
    )


def calculate_confidence(score: float, max_score: float, rules: dict[str, Any]) -> str:
    """Convert a raw score to a qualitative confidence label."""

    if max_score <= 0:
        return str(rules.get("low_label", "LOW"))

    ratio = score / max_score

    if ratio >= float(rules.get("high_min_ratio", 0.70)):
        return str(rules.get("high_label", "HIGH"))
    if ratio >= float(rules.get("medium_min_ratio", 0.40)):
        return str(rules.get("medium_label", "MEDIUM"))
    return str(rules.get("low_label", "LOW"))


def select_output_columns(df: pd.DataFrame, output_columns: list[str]) -> pd.DataFrame:
    """Return only the configured columns that are present in the frame."""

    if not output_columns:
        return df
    return df[[column for column in output_columns if column in df.columns]]
