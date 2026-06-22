from pathlib import Path
import argparse
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OPERATIONAL_HISTORY_FILE = PROCESSED_DIR / "weekly_operational_usage_all.csv"
WEEKLY_SUMMARY_FILE = PROCESSED_DIR / "weekly_summary_all.csv"


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )
    return df


def find_column(df: pd.DataFrame, required_terms: list[str], fallback_candidates: list[str]) -> str:
    for candidate in fallback_candidates:
        if candidate in df.columns:
            return candidate

    for col in df.columns:
        if all(term in col for term in required_terms):
            return col

    raise ValueError(
        f"Could not identify column with terms {required_terms}. "
        f"Columns found: {list(df.columns)}"
    )


def find_optional_column(
    df: pd.DataFrame,
    required_terms: list[str],
    fallback_candidates: list[str],
) -> str | None:
    try:
        return find_column(df, required_terms, fallback_candidates)
    except ValueError:
        return None


def infer_week_dates_from_filename(file_path: Path) -> tuple[str | None, str | None]:
    """
    Tries to infer week dates from filenames like:
    Brookhaven National Lab users export (2026-06-01 - 2026-06-08).csv
    """
    name = file_path.stem

    import re

    match = re.search(r"(\d{4}-\d{2}-\d{2})\s*-\s*(\d{4}-\d{2}-\d{2})", name)
    if match:
        return match.group(1), match.group(2)

    return None, None


def load_weekly_csv(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find weekly CSV: {file_path}")

    df = pd.read_csv(file_path)
    df = clean_column_names(df)

    email_col = find_column(
        df,
        required_terms=["email"],
        fallback_candidates=["email", "user_email", "email_address"],
    )

    credit_col = find_column(
        df,
        required_terms=["credit"],
        fallback_candidates=[
            "credits_used",
            "credit_used",
            "total_credits_used",
            "credits",
            "amu_credits",
        ],
    )

    message_col = find_optional_column(
        df,
        required_terms=["message"],
        fallback_candidates=["messages", "total_messages", "message_count"],
    )

    df = df.rename(
        columns={
            email_col: "email",
            credit_col: "credits_used",
        }
    )

    if message_col:
        df = df.rename(columns={message_col: "messages"})
    else:
        df["messages"] = 0

    df["credits_used"] = pd.to_numeric(df["credits_used"], errors="coerce").fillna(0)
    df["messages"] = pd.to_numeric(df["messages"], errors="coerce").fillna(0)

    df["is_credit_active"] = df["credits_used"] > 0
    df["is_message_active"] = df["messages"] > 0

    return df


def add_week_metadata(
    df: pd.DataFrame,
    file_path: Path,
    week_start: str | None,
    week_end: str | None,
) -> pd.DataFrame:
    df = df.copy()

    inferred_start, inferred_end = infer_week_dates_from_filename(file_path)

    week_start = week_start or inferred_start
    week_end = week_end or inferred_end

    if not week_start or not week_end:
        raise ValueError(
            "Could not infer week dates from filename. "
            "Please pass --week-start YYYY-MM-DD and --week-end YYYY-MM-DD."
        )

    df["week_start"] = pd.to_datetime(week_start)
    df["week_end"] = pd.to_datetime(week_end)
    df["source_file"] = file_path.name

    return df


def build_week_summary(df: pd.DataFrame) -> pd.DataFrame:
    credit_active = df[df["is_credit_active"]]

    summary = {
        "week_start": df["week_start"].iloc[0],
        "week_end": df["week_end"].iloc[0],
        "source_file": df["source_file"].iloc[0],
        "reported_rows": len(df),
        "unique_users": df["email"].nunique(),
        "message_active_users": int(df["is_message_active"].sum()),
        "credit_active_users": int(df["is_credit_active"].sum()),
        "total_credits_used": float(df["credits_used"].sum()),
        "total_messages": float(df["messages"].sum()),
        "avg_credits_per_credit_active_user": float(credit_active["credits_used"].mean())
        if len(credit_active) > 0
        else 0,
        "median_credits_per_credit_active_user": float(credit_active["credits_used"].median())
        if len(credit_active) > 0
        else 0,
        "p95_credits_per_credit_active_user": float(credit_active["credits_used"].quantile(0.95))
        if len(credit_active) > 0
        else 0,
    }

    return pd.DataFrame([summary])


def append_without_duplicate_weeks(new_df: pd.DataFrame, output_file: Path) -> pd.DataFrame:
    if output_file.exists():
        existing = pd.read_csv(output_file)
        existing["week_start"] = pd.to_datetime(existing["week_start"])
        existing["week_end"] = pd.to_datetime(existing["week_end"])

        new_week_start = pd.to_datetime(new_df["week_start"].iloc[0])
        existing = existing[existing["week_start"] != new_week_start]

        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    combined = combined.sort_values(["week_start", "email"] if "email" in combined.columns else ["week_start"])
    return combined


def save_outputs(cleaned_week: pd.DataFrame, week_summary: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    operational_history = append_without_duplicate_weeks(
        cleaned_week,
        OPERATIONAL_HISTORY_FILE,
    )

    summary_history = append_without_duplicate_weeks(
        week_summary,
        WEEKLY_SUMMARY_FILE,
    )

    operational_history.to_csv(OPERATIONAL_HISTORY_FILE, index=False)
    summary_history.to_csv(WEEKLY_SUMMARY_FILE, index=False)

    latest_cleaned_file = PROCESSED_DIR / "latest_week_operational_usage_cleaned.csv"
    latest_summary_file = PROCESSED_DIR / "latest_week_summary.csv"

    cleaned_week.to_csv(latest_cleaned_file, index=False)
    week_summary.to_csv(latest_summary_file, index=False)


def print_summary(week_summary: pd.DataFrame) -> None:
    row = week_summary.iloc[0]

    print("\nWeekly ingestion complete")
    print("-------------------------")
    print(f"Week:                       {row['week_start']} to {row['week_end']}")
    print(f"Source file:                {row['source_file']}")
    print(f"Reported rows:              {row['reported_rows']:,}")
    print(f"Unique users:               {row['unique_users']:,}")
    print(f"Message-active users:       {row['message_active_users']:,}")
    print(f"Credit-active users:        {row['credit_active_users']:,}")
    print(f"Total credits used:         {row['total_credits_used']:,.2f}")
    print(f"Total messages:             {row['total_messages']:,.0f}")
    print(f"Avg credits/credit user:    {row['avg_credits_per_credit_active_user']:,.2f}")
    print(f"Median credits/credit user: {row['median_credits_per_credit_active_user']:,.2f}")
    print(f"P95 credits/credit user:    {row['p95_credits_per_credit_active_user']:,.2f}")

    print("\nFiles updated")
    print("-------------")
    print(OPERATIONAL_HISTORY_FILE)
    print(WEEKLY_SUMMARY_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a weekly OpenAI AMU usage CSV.")
    parser.add_argument("--file", required=True, help="Path to weekly CSV file.")
    parser.add_argument("--week-start", required=False, help="Week start date, YYYY-MM-DD.")
    parser.add_argument("--week-end", required=False, help="Week end date, YYYY-MM-DD.")

    args = parser.parse_args()

    file_path = Path(args.file)

    df = load_weekly_csv(file_path)
    df = add_week_metadata(df, file_path, args.week_start, args.week_end)

    week_summary = build_week_summary(df)

    save_outputs(df, week_summary)
    print_summary(week_summary)


if __name__ == "__main__":
    main()