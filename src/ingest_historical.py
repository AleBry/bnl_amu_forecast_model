from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

HISTORICAL_FILE = PROJECT_ROOT / "data" / "historical" / "bnl_weekly_user_export_final.xlsx"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EXPECTED_TOTAL_CREDITS = 375750.68852


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def load_historical_usage() -> pd.DataFrame:
    if not HISTORICAL_FILE.exists():
        raise FileNotFoundError(f"Could not find historical file: {HISTORICAL_FILE}")

    df = pd.read_excel(HISTORICAL_FILE, sheet_name="weekly_user_export")
    df = clean_column_names(df)

    required_columns = [
        "period_start",
        "period_end",
        "email",
        "messages",
        "credits_used",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["period_start"] = pd.to_datetime(df["period_start"])
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["credits_used"] = pd.to_numeric(df["credits_used"], errors="coerce").fillna(0)
    df["messages"] = pd.to_numeric(df["messages"], errors="coerce").fillna(0)

    df["is_credit_active"] = df["credits_used"] > 0
    df["is_message_active"] = df["messages"] > 0

    return df


def build_weekly_summary(df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        df.groupby(["period_start", "period_end"], as_index=False)
        .agg(
            total_credits_used=("credits_used", "sum"),
            total_messages=("messages", "sum"),
            credit_active_users=("is_credit_active", "sum"),
            message_active_users=("is_message_active", "sum"),
            unique_users=("email", "nunique"),
        )
        .sort_values("period_start")
    )

    weekly["credits_per_credit_active_user"] = (
        weekly["total_credits_used"] / weekly["credit_active_users"]
    )

    return weekly


def build_user_summary(df: pd.DataFrame) -> pd.DataFrame:
    user_summary = (
        df.groupby("email", as_index=False)
        .agg(
            user_total_credits=("credits_used", "sum"),
            user_total_messages=("messages", "sum"),
            active_credit_weeks=("is_credit_active", "sum"),
            active_message_weeks=("is_message_active", "sum"),
            first_period=("period_start", "min"),
            last_period=("period_start", "max"),
        )
        .sort_values("user_total_credits", ascending=False)
    )

    return user_summary


def validate_totals(df: pd.DataFrame) -> None:
    actual_total = df["credits_used"].sum()
    difference = abs(actual_total - EXPECTED_TOTAL_CREDITS)

    print("\nValidation")
    print("----------")
    print(f"Expected total credits: {EXPECTED_TOTAL_CREDITS:,.2f}")
    print(f"Actual total credits:   {actual_total:,.2f}")
    print(f"Difference:             {difference:,.6f}")

    if difference > 0.01:
        raise ValueError("Historical total does not match expected total.")

    print("PASS: Historical total matches expected total.")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_historical_usage()
    weekly_summary = build_weekly_summary(df)
    user_summary = build_user_summary(df)

    validate_totals(df)

    cleaned_path = PROCESSED_DIR / "historical_usage_cleaned.csv"
    weekly_path = PROCESSED_DIR / "historical_weekly_summary.csv"
    user_path = PROCESSED_DIR / "historical_user_summary.csv"

    df.to_csv(cleaned_path, index=False)
    weekly_summary.to_csv(weekly_path, index=False)
    user_summary.to_csv(user_path, index=False)

    print("\nHistorical Summary")
    print("------------------")
    print(f"Rows loaded:              {len(df):,}")
    print(f"Unique users:             {df['email'].nunique():,}")
    print(f"Weekly periods:           {df['period_start'].nunique():,}")
    print(f"Total credits used:       {df['credits_used'].sum():,.2f}")
    print(f"Average weekly burn:      {weekly_summary['total_credits_used'].mean():,.2f}")
    print(f"Median weekly burn:       {weekly_summary['total_credits_used'].median():,.2f}")

    print("\nFiles written")
    print("-------------")
    print(cleaned_path)
    print(weekly_path)
    print(user_path)


if __name__ == "__main__":
    main()