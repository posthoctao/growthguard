from pathlib import Path
from pprint import pprint
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BOUNCE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "bounce_rate_clean.csv"
)

CONVERSION_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "conversion_funnel_clean.csv"
)


def load_bounce_data() -> pd.DataFrame:
    """Load and validate cleaned bounce rate data."""

    if not BOUNCE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned bounce rate data was not found: {BOUNCE_DATA_PATH}"
        )

    df = pd.read_csv(BOUNCE_DATA_PATH)

    required_columns = {
        "month",
        "bounce_rate",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in bounce data: {sorted(missing_columns)}"
        )

    df["month"] = pd.to_datetime(
        df["month"],
        errors="coerce",
    )

    df["bounce_rate"] = pd.to_numeric(
        df["bounce_rate"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "month",
            "bounce_rate",
        ]
    )

    return df


def load_conversion_funnel_data() -> pd.DataFrame:
    """Load and validate cleaned conversion funnel data."""

    if not CONVERSION_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned conversion funnel data was not found: {CONVERSION_DATA_PATH}"
        )

    df = pd.read_csv(CONVERSION_DATA_PATH)

    required_columns = {
        "month",
        "sessions",
        "cart_addition_rate",
        "reached_checkout_rate",
        "checkout_completion_rate",
        "conversion_rate",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in conversion data: {sorted(missing_columns)}"
        )

    df["month"] = pd.to_datetime(
        df["month"],
        errors="coerce",
    )

    numeric_columns = [
        "sessions",
        "cart_addition_rate",
        "reached_checkout_rate",
        "checkout_completion_rate",
        "conversion_rate",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "month",
            "sessions",
            "cart_addition_rate",
            "reached_checkout_rate",
            "checkout_completion_rate",
            "conversion_rate",
        ]
    )

    return df


def build_funnel_data() -> pd.DataFrame:
    """Merge bounce rate data with conversion funnel data."""

    bounce_df = load_bounce_data()
    conversion_df = load_conversion_funnel_data()

    funnel_df = conversion_df.merge(
        bounce_df,
        on="month",
        how="left",
    )

    required_columns = [
        "month",
        "sessions",
        "bounce_rate",
        "cart_addition_rate",
        "reached_checkout_rate",
        "checkout_completion_rate",
        "conversion_rate",
    ]

    funnel_df = funnel_df.dropna(
        subset=required_columns
    )

    funnel_df = funnel_df.sort_values(
        "month"
    ).reset_index(drop=True)

    return funnel_df


def judge_metric_status(
    metric: str,
    change: float,
) -> str:
    """
    Judge whether a metric improved or worsened.

    For bounce rate, lower is better.
    For other funnel metrics, higher is better.
    """

    if change == 0:
        return "no_change"

    if metric == "bounce_rate":
        if change > 0:
            return "worsened"
        return "improved"

    if change > 0:
        return "improved"

    return "worsened"


def analyze_latest_funnel_change() -> dict[str, Any]:
    """
    Compare the latest two complete months and summarize
    funnel metric changes.
    """

    funnel_df = build_funnel_data()

    current_month = (
        pd.Timestamp.today()
        .to_period("M")
        .to_timestamp()
    )

    complete_df = funnel_df[
        funnel_df["month"] < current_month
    ].copy()

    available_months = sorted(
        complete_df["month"].unique()
    )

    if len(available_months) < 2:
        raise ValueError(
            "At least two complete months are required for funnel analysis."
        )

    previous_month = pd.Timestamp(available_months[-2])
    latest_month = pd.Timestamp(available_months[-1])

    previous_row = complete_df[
        complete_df["month"] == previous_month
    ].iloc[0]

    latest_row = complete_df[
        complete_df["month"] == latest_month
    ].iloc[0]

    funnel_metrics = [
        "sessions",
        "bounce_rate",
        "cart_addition_rate",
        "reached_checkout_rate",
        "checkout_completion_rate",
        "conversion_rate",
    ]

    metric_changes = []

    for metric in funnel_metrics:
        previous_value = float(previous_row[metric])
        latest_value = float(latest_row[metric])
        change = latest_value - previous_value

        if previous_value == 0:
            change_pct = None
        else:
            change_pct = change / previous_value * 100

        metric_changes.append(
            {
                "metric": metric,
                "previous_value": round(previous_value, 6),
                "latest_value": round(latest_value, 6),
                "change": round(change, 6),
                "change_pct": (
                    None
                    if change_pct is None
                    else round(change_pct, 2)
                ),
                "status": judge_metric_status(
                    metric=metric,
                    change=change,
                ),
            }
        )

    worsened_metrics = [
        item["metric"]
        for item in metric_changes
        if item["status"] == "worsened"
    ]

    improved_metrics = [
        item["metric"]
        for item in metric_changes
        if item["status"] == "improved"
    ]

    return {
        "previous_month": previous_month.strftime("%Y-%m"),
        "latest_month": latest_month.strftime("%Y-%m"),
        "metric_changes": metric_changes,
        "worsened_metrics": worsened_metrics,
        "improved_metrics": improved_metrics,
        "summary": (
            "Funnel traffic increased or decreased based on sessions, "
            "while funnel quality is judged by bounce rate, cart addition rate, "
            "reached checkout rate, checkout completion rate, and conversion rate."
        ),
    }


if __name__ == "__main__":
    result = analyze_latest_funnel_change()
    pprint(result, sort_dicts=False)