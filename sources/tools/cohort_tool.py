from pathlib import Path
from pprint import pprint
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "cohort_subscriber_retention_clean.csv"
)

EARLY_LIFECYCLE_MONTHS = [1, 2, 3]
DEFAULT_COMPARISON_WINDOW = 3


def load_cohort_retention_data() -> pd.DataFrame:
    """
    Load cleaned subscriber cohort retention data.

    Retention rates are calculated using Month 0 as the denominator:
    retained_subscribers_month_N / retained_subscribers_month_0
    """

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "cohort_month",
        "cohort_subscribers",
        "retained_subscribers_month_0",
        "retained_subscribers_month_1",
        "retained_subscribers_month_2",
        "retained_subscribers_month_3",
        "latest_observed_lifecycle_month",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required cohort retention columns: "
            f"{missing_columns}"
        )

    df["cohort_month"] = pd.to_datetime(
        df["cohort_month"],
        errors="coerce",
    )

    df["cohort_subscribers"] = pd.to_numeric(
        df["cohort_subscribers"],
        errors="coerce",
    )

    df["latest_observed_lifecycle_month"] = pd.to_numeric(
        df["latest_observed_lifecycle_month"],
        errors="coerce",
    ).astype("Int64")

    retained_columns = [
        "retained_subscribers_month_0",
        "retained_subscribers_month_1",
        "retained_subscribers_month_2",
        "retained_subscribers_month_3",
    ]

    for column in retained_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    month_0_base = df[
        "retained_subscribers_month_0"
    ].replace(0, pd.NA)

    for lifecycle_month in EARLY_LIFECYCLE_MONTHS:
        retained_column = (
            f"retained_subscribers_month_{lifecycle_month}"
        )

        rate_column = (
            f"retention_rate_month_{lifecycle_month}"
        )

        df[rate_column] = (
            df[retained_column]
            / month_0_base
        )

    df = (
        df.dropna(
            subset=[
                "cohort_month",
                "cohort_subscribers",
                "retained_subscribers_month_0",
            ]
        )
        .drop_duplicates(subset=["cohort_month"])
        .sort_values("cohort_month")
        .reset_index(drop=True)
    )

    return df


def get_mature_cohorts(
    df: pd.DataFrame,
    required_lifecycle_month: int = 3,
) -> pd.DataFrame:
    """
    Keep only cohorts with complete Month 1 to Month 3 retention data.

    Recent cohorts that have not reached Month 3 are excluded instead
    of being treated as zero retention.
    """

    required_rate_columns = [
        f"retention_rate_month_{month}"
        for month in range(1, required_lifecycle_month + 1)
    ]

    mature_cohorts = df.loc[
        (
            df["latest_observed_lifecycle_month"]
            >= required_lifecycle_month
        )
        & df[required_rate_columns].notna().all(axis=1)
    ].copy()

    if mature_cohorts.empty:
        raise ValueError(
            "No mature cohorts are available for early retention analysis."
        )

    return (
        mature_cohorts
        .sort_values("cohort_month")
        .reset_index(drop=True)
    )


def get_comparison_groups(
    mature_cohorts: pd.DataFrame,
    comparison_window: int = DEFAULT_COMPARISON_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare the latest mature cohorts with the immediately previous cohorts.

    Example:
    latest 3 mature cohorts vs. the 3 mature cohorts immediately before them.
    """

    group_size = min(
        comparison_window,
        len(mature_cohorts) // 2,
    )

    if group_size < 1:
        raise ValueError(
            "Not enough mature cohorts for a retention comparison."
        )

    previous_cohorts = mature_cohorts.iloc[
        -2 * group_size:-group_size
    ].copy()

    recent_cohorts = mature_cohorts.iloc[
        -group_size:
    ].copy()

    return previous_cohorts, recent_cohorts


def calculate_average_retention_pct(
    cohort_df: pd.DataFrame,
) -> dict[str, float]:
    """Calculate average Month 1 to Month 3 retention percentages."""

    result = {}

    for lifecycle_month in EARLY_LIFECYCLE_MONTHS:
        rate_column = (
            f"retention_rate_month_{lifecycle_month}"
        )

        result[f"month_{lifecycle_month}"] = round(
            float(cohort_df[rate_column].mean() * 100),
            2,
        )

    return result


def calculate_retention_change_percentage_points(
    previous_retention: dict[str, float],
    recent_retention: dict[str, float],
) -> dict[str, float]:
    """
    Calculate recent-minus-previous retention changes in percentage points.
    """

    return {
        month_name: round(
            float(
                recent_retention[month_name]
                - previous_retention[month_name]
            ),
            2,
        )
        for month_name in recent_retention
    }


def judge_early_retention_status(
    retention_change_pct_points: dict[str, float],
) -> str:
    """
    Classify early retention trend.

    A change within +/- 0.5 percentage points is treated as stable.
    """

    positive_changes = sum(
        change > 0.5
        for change in retention_change_pct_points.values()
    )

    negative_changes = sum(
        change < -0.5
        for change in retention_change_pct_points.values()
    )

    if negative_changes > positive_changes:
        return "worsened"

    if positive_changes > negative_changes:
        return "improved"

    return "mixed"


def find_worst_month_3_cohort(
    mature_cohorts: pd.DataFrame,
) -> dict[str, Any]:
    """Find the mature cohort with the lowest Month 3 retention rate."""

    worst_row = mature_cohorts.sort_values(
        "retention_rate_month_3",
        ascending=True,
    ).iloc[0]

    return {
        "cohort_month": worst_row[
            "cohort_month"
        ].strftime("%Y-%m"),
        "cohort_subscribers": int(
            worst_row["cohort_subscribers"]
        ),
        "month_0_subscribers": int(
            worst_row["retained_subscribers_month_0"]
        ),
        "month_1_retention_pct": round(
            float(worst_row["retention_rate_month_1"] * 100),
            2,
        ),
        "month_2_retention_pct": round(
            float(worst_row["retention_rate_month_2"] * 100),
            2,
        ),
        "month_3_retention_pct": round(
            float(worst_row["retention_rate_month_3"] * 100),
            2,
        ),
    }


def find_largest_early_dropoff(
    recent_retention_pct: dict[str, float],
) -> dict[str, Any]:
    """
    Find the largest retention drop from Month 0 to Month 3.

    Month 0 is treated as a 100% baseline because all later retention
    rates are calculated using retained_subscribers_month_0.
    """

    retention_path = {
        "month_0": 100.0,
        "month_1": float(recent_retention_pct["month_1"]),
        "month_2": float(recent_retention_pct["month_2"]),
        "month_3": float(recent_retention_pct["month_3"]),
    }

    transitions = []

    for previous_month, latest_month in [
        ("month_0", "month_1"),
        ("month_1", "month_2"),
        ("month_2", "month_3"),
    ]:
        drop_pct_points = round(
            float(
                retention_path[previous_month]
                - retention_path[latest_month]
            ),
            2,
        )

        transitions.append(
            {
                "from": previous_month,
                "to": latest_month,
                "drop_percentage_points": drop_pct_points,
            }
        )

    return max(
        transitions,
        key=lambda item: item["drop_percentage_points"],
    )


def get_cohort_labels(
    cohort_df: pd.DataFrame,
) -> list[str]:
    """Return cohort months as YYYY-MM labels."""

    return [
        month.strftime("%Y-%m")
        for month in cohort_df["cohort_month"]
    ]


def analyze_early_cohort_retention() -> dict[str, Any]:
    """
    Analyze early subscriber retention using mature cohorts only.

    The tool compares the latest mature cohorts with the immediately
    previous mature cohorts and focuses on Month 1 to Month 3 retention.
    """

    df = load_cohort_retention_data()

    mature_cohorts = get_mature_cohorts(
        df,
        required_lifecycle_month=3,
    )

    previous_cohorts, recent_cohorts = get_comparison_groups(
        mature_cohorts,
        comparison_window=DEFAULT_COMPARISON_WINDOW,
    )

    previous_retention_pct = calculate_average_retention_pct(
        previous_cohorts
    )

    recent_retention_pct = calculate_average_retention_pct(
        recent_cohorts
    )

    retention_change_pct_points = (
        calculate_retention_change_percentage_points(
            previous_retention=previous_retention_pct,
            recent_retention=recent_retention_pct,
        )
    )

    early_retention_status = judge_early_retention_status(
        retention_change_pct_points
    )

    worst_month_3_cohort = find_worst_month_3_cohort(
        mature_cohorts
    )

    largest_early_dropoff = find_largest_early_dropoff(
        recent_retention_pct
    )

    return {
        "analysis_scope": {
            "mature_cohort_count": int(len(mature_cohorts)),
            "latest_mature_cohort": mature_cohorts[
                "cohort_month"
            ].max().strftime("%Y-%m"),
            "retention_denominator": (
                "retained_subscribers_month_0"
            ),
            "comparison_method": (
                "Latest mature cohorts compared with the immediately "
                "previous mature cohorts."
            ),
        },
        "early_retention_status": early_retention_status,
        "comparison_groups": {
            "previous_cohorts": get_cohort_labels(
                previous_cohorts
            ),
            "recent_cohorts": get_cohort_labels(
                recent_cohorts
            ),
        },
        "previous_average_retention_pct": (
            previous_retention_pct
        ),
        "recent_average_retention_pct": (
            recent_retention_pct
        ),
        "retention_change_percentage_points": (
            retention_change_pct_points
        ),
        "worst_month_3_cohort": worst_month_3_cohort,
        "largest_early_dropoff_in_recent_cohorts": (
            largest_early_dropoff
        ),
    }


if __name__ == "__main__":
    result = analyze_early_cohort_retention()
    pprint(result, sort_dicts=False)