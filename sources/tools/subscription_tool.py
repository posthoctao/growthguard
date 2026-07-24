from pathlib import Path
from pprint import pprint

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "subscription_summary_clean.csv"
)


def load_subscription_data() -> pd.DataFrame:
    """Load and validate cleaned subscription summary data."""

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "date",
        "month",
        "active_subscribers",
        "new_subscribers",
        "resumed_subscribers",
        "reactivated_subscribers",
        "paused_subscribers",
        "churned_subscribers",
        "expired_subscribers",
        "activated_subscribers",
        "deactivated_subscribers",
        "net_subscriber_change",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required subscription columns: {missing_columns}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df["month"] = pd.to_datetime(
        df["month"],
        errors="coerce",
    )

    numeric_columns = [
        "active_subscribers",
        "new_subscribers",
        "resumed_subscribers",
        "reactivated_subscribers",
        "paused_subscribers",
        "churned_subscribers",
        "expired_subscribers",
        "activated_subscribers",
        "deactivated_subscribers",
        "net_subscriber_change",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.dropna(subset=["date", "month"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    return df


def build_monthly_subscription_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert daily subscription data into monthly subscription metrics.

    Flow metrics are summed within each month.
    Active subscribers use the last available date of each month.
    """

    flow_columns = [
        "new_subscribers",
        "resumed_subscribers",
        "reactivated_subscribers",
        "paused_subscribers",
        "churned_subscribers",
        "expired_subscribers",
        "activated_subscribers",
        "deactivated_subscribers",
        "net_subscriber_change",
    ]

    monthly_flows = (
        df.groupby("month", as_index=False)[flow_columns]
        .sum()
    )

    sorted_df = df.sort_values("date").copy()

    monthly_start_active = (
        sorted_df.groupby("month", as_index=False)[
            "active_subscribers"
        ]
        .first()
        .rename(
            columns={
                "active_subscribers": "month_start_active_subscribers"
            }
        )
    )

    monthly_end_active = (
        sorted_df.groupby("month", as_index=False)[
            "active_subscribers"
        ]
        .last()
        .rename(
            columns={
                "active_subscribers": "month_end_active_subscribers"
            }
        )
    )

    monthly_summary = (
        monthly_flows
        .merge(
            monthly_start_active,
            on="month",
            how="left",
        )
        .merge(
            monthly_end_active,
            on="month",
            how="left",
        )
        .sort_values("month")
        .reset_index(drop=True)
    )

    monthly_summary["estimated_deactivation_rate"] = (
        monthly_summary["deactivated_subscribers"]
        / monthly_summary[
            "month_start_active_subscribers"
        ].replace(0, pd.NA)
    )

    monthly_summary["estimated_churn_rate"] = (
        monthly_summary["churned_subscribers"]
        / monthly_summary[
            "month_start_active_subscribers"
        ].replace(0, pd.NA)
    )

    return monthly_summary


def get_latest_two_complete_months(
    monthly_summary: pd.DataFrame,
    last_data_date: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return the latest two complete months.

    The final month is excluded when the source data ends before
    that calendar month is complete.
    """

    available_months = sorted(
        monthly_summary["month"].dropna().unique()
    )

    latest_data_month = last_data_date.to_period("M")
    latest_month_end = (
        last_data_date + pd.offsets.MonthEnd(0)
    ).normalize()

    is_latest_month_complete = (
        last_data_date.normalize() == latest_month_end
    )

    if not is_latest_month_complete:
        available_months = [
            month
            for month in available_months
            if pd.Timestamp(month).to_period("M")
            != latest_data_month
        ]

    if len(available_months) < 2:
        raise ValueError(
            "Not enough complete months to compare subscription performance."
        )

    previous_month = pd.Timestamp(available_months[-2])
    latest_month = pd.Timestamp(available_months[-1])

    return previous_month, latest_month


def percentage_change(
    previous_value: float,
    latest_value: float,
) -> float | None:
    """Calculate percentage change safely."""

    if pd.isna(previous_value) or previous_value == 0:
        return None

    return round(
        (latest_value - previous_value)
        / previous_value
        * 100,
        2,
    )


def judge_subscription_status(
    active_subscriber_change: float,
    churned_subscriber_change: float,
    deactivation_rate_change: float | None,
) -> str:
    """Return a simple subscription-health interpretation."""

    if (
        active_subscriber_change > 0
        and churned_subscriber_change <= 0
    ):
        return "improved"

    if active_subscriber_change < 0:
        return "worsened"

    if churned_subscriber_change > 0:
        return "worsened"

    if (
        deactivation_rate_change is not None
        and deactivation_rate_change > 0
    ):
        return "worsened"

    return "mixed"


def analyze_latest_subscription_change() -> dict:
    """
    Compare the latest two complete months of subscription performance.
    """

    df = load_subscription_data()

    monthly_summary = build_monthly_subscription_summary(df)

    previous_month, latest_month = (
        get_latest_two_complete_months(
            monthly_summary=monthly_summary,
            last_data_date=df["date"].max(),
        )
    )

    previous_row = monthly_summary.loc[
        monthly_summary["month"] == previous_month
    ].iloc[0]

    latest_row = monthly_summary.loc[
        monthly_summary["month"] == latest_month
    ].iloc[0]

    previous_active = float(
        previous_row["month_end_active_subscribers"]
    )
    latest_active = float(
        latest_row["month_end_active_subscribers"]
    )

    active_change = latest_active - previous_active

    previous_churned = float(
        previous_row["churned_subscribers"]
    )
    latest_churned = float(
        latest_row["churned_subscribers"]
    )

    churned_change = latest_churned - previous_churned

    previous_deactivation_rate = float(
        previous_row["estimated_deactivation_rate"]
    )
    latest_deactivation_rate = float(
        latest_row["estimated_deactivation_rate"]
    )

    deactivation_rate_change = (
        latest_deactivation_rate
        - previous_deactivation_rate
    )

    subscription_status = judge_subscription_status(
        active_subscriber_change=active_change,
        churned_subscriber_change=churned_change,
        deactivation_rate_change=deactivation_rate_change,
    )

    return {
        "previous_month": previous_month.strftime("%Y-%m"),
        "latest_month": latest_month.strftime("%Y-%m"),
        "subscription_status": subscription_status,
        "active_subscribers": {
            "previous": round(previous_active, 0),
            "latest": round(latest_active, 0),
            "change": round(active_change, 0),
            "change_pct": percentage_change(
                previous_active,
                latest_active,
            ),
        },
        "new_subscribers": {
            "previous": round(
                float(previous_row["new_subscribers"]),
                0,
            ),
            "latest": round(
                float(latest_row["new_subscribers"]),
                0,
            ),
            "change": round(
                float(latest_row["new_subscribers"])
                - float(previous_row["new_subscribers"]),
                0,
            ),
        },
        "churned_subscribers": {
            "previous": round(previous_churned, 0),
            "latest": round(latest_churned, 0),
            "change": round(churned_change, 0),
        },
        "deactivated_subscribers": {
            "previous": round(
                float(previous_row["deactivated_subscribers"]),
                0,
            ),
            "latest": round(
                float(latest_row["deactivated_subscribers"]),
                0,
            ),
            "change": round(
                float(latest_row["deactivated_subscribers"])
                - float(previous_row["deactivated_subscribers"]),
                0,
            ),
        },
        "estimated_deactivation_rate": {
            "previous": round(
                previous_deactivation_rate,
                4,
            ),
            "latest": round(
                latest_deactivation_rate,
                4,
            ),
            "change": round(
                deactivation_rate_change,
                4,
            ),
        },
        "net_subscriber_change": {
            "previous": round(
                float(previous_row["net_subscriber_change"]),
                0,
            ),
            "latest": round(
                float(latest_row["net_subscriber_change"]),
                0,
            ),
            "change": round(
                float(latest_row["net_subscriber_change"])
                - float(previous_row["net_subscriber_change"]),
                0,
            ),
        },
    }


if __name__ == "__main__":
    result = analyze_latest_subscription_change()
    pprint(result, sort_dicts=False)