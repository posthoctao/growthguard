from __future__ import annotations

from pathlib import Path
from pprint import pprint
from typing import Any

import pandas as pd


# ============================================================
# Paths and settings
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "sales_by_channel_clean.csv"
)

# ============================================================
# Data loading
# ============================================================

def load_sales_data() -> pd.DataFrame:
    """
    Load and validate cleaned monthly sales-by-channel data.

    Required columns:
    - month
    - sales_channel
    - total_sales
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned sales data was not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    df = df.rename(
        columns={
            "total_sales_$": "total_sales",
        }
    )

    required_columns = {
        "month",
        "sales_channel",
        "total_sales",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required sales columns: "
            f"{sorted(missing_columns)}"
        )

    df["month"] = pd.to_datetime(
        df["month"],
        errors="coerce",
    )

    df["month"] = (
        df["month"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    df["sales_channel"] = (
        df["sales_channel"]
        .astype(str)
        .str.strip()
    )

    df["total_sales"] = pd.to_numeric(
        df["total_sales"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "month",
            "sales_channel",
            "total_sales",
        ]
    )

    df = df[
        df["sales_channel"] != ""
    ].copy()

    if df.empty:
        raise ValueError(
            "Sales data is empty after cleaning."
        )

    return df


# ============================================================
# Month selection
# ============================================================


def normalize_end_month(
    end_month: str,
) -> pd.Timestamp:
    """
    Convert YYYY-MM text into a normalized month-start timestamp.
    """
    if not isinstance(end_month, str):
        raise ValueError(
            "end_month must be a string in YYYY-MM format."
        )

    try:
        return pd.Period(
            end_month.strip(),
            freq="M",
        ).to_timestamp()

    except ValueError as error:
        raise ValueError(
            "end_month must use YYYY-MM format, "
            f"received: {end_month}"
        ) from error


def get_current_month_start() -> pd.Timestamp:
    """Return the first day of the current calendar month."""
    return (
        pd.Timestamp.today()
        .to_period("M")
        .to_timestamp()
    )


def select_comparison_months(
    df: pd.DataFrame,
    end_month: str | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    """
    Select two consecutive completed calendar months.

    Rules:
    - The current calendar month is automatically excluded.
    - When end_month is None, use the latest two completed months.
    - When end_month is provided, compare it with its prior
      calendar month.
    """
    current_month = get_current_month_start()

    complete_df = df[
        df["month"] < current_month
    ].copy()

    available_months = sorted(
        pd.Timestamp(month)
        for month in complete_df["month"].unique()
    )

    if len(available_months) < 2:
        raise ValueError(
            "At least two complete sales months are required."
        )

    if end_month is None:
        previous_month = available_months[-2]
        latest_month = available_months[-1]

        return (
            previous_month,
            latest_month,
            "latest_complete_months",
        )

    latest_month = normalize_end_month(end_month)

    if latest_month >= current_month:
        raise ValueError(
            f"{end_month} is not a completed calendar month."
        )

    if latest_month not in available_months:
        raise ValueError(
            f"{end_month} is not available in the sales dataset."
        )

    previous_month = (
        latest_month - pd.DateOffset(months=1)
    )

    if previous_month not in available_months:
        raise ValueError(
            "The previous complete month is unavailable for "
            f"{end_month}. Expected: "
            f"{previous_month.strftime('%Y-%m')}."
        )

    return (
        previous_month,
        latest_month,
        "explicit_complete_month",
    )

# ============================================================
# Analytics helpers
# ============================================================

def aggregate_sales_by_channel(
    df: pd.DataFrame,
    month: pd.Timestamp,
    output_column: str,
) -> pd.DataFrame:
    """Aggregate sales by channel for one month."""
    return (
        df[
            df["month"] == month
        ]
        .groupby(
            "sales_channel",
            as_index=False,
        )["total_sales"]
        .sum()
        .rename(
            columns={
                "total_sales": output_column,
            }
        )
    )


def format_channel_rows(
    result_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Format channel rows for a JSON-friendly result."""
    rows: list[dict[str, Any]] = []

    for _, row in result_df.iterrows():
        previous_sales = float(
            row["previous_sales"]
        )

        latest_sales = float(
            row["latest_sales"]
        )

        sales_change = float(
            row["sales_change"]
        )

        sales_change_pct = (
            None
            if previous_sales == 0
            else sales_change / previous_sales * 100
        )

        rows.append(
            {
                "sales_channel": str(
                    row["sales_channel"]
                ),
                "previous_sales": round(
                    previous_sales,
                    2,
                ),
                "latest_sales": round(
                    latest_sales,
                    2,
                ),
                "sales_change": round(
                    sales_change,
                    2,
                ),
                "sales_change_pct": (
                    None
                    if sales_change_pct is None
                    else round(
                        sales_change_pct,
                        2,
                    )
                ),
            }
        )

    return rows


# ============================================================
# Main analysis
# ============================================================

def analyze_sales_change(
    end_month: str | None = None,
) -> dict[str, Any]:
    """
    Compare sales between two consecutive complete months.

    Args:
        end_month:
            Optional target month in YYYY-MM format.

            Example:
            end_month="2026-05"
            compares 2026-04 with 2026-05.

            When None, uses the latest two completed calendar months.
    """
    df = load_sales_data()

    (
        previous_month,
        latest_month,
        selection_mode,
    ) = select_comparison_months(
        df=df,
        end_month=end_month,
    )

    previous_data = aggregate_sales_by_channel(
        df=df,
        month=previous_month,
        output_column="previous_sales",
    )

    latest_data = aggregate_sales_by_channel(
        df=df,
        month=latest_month,
        output_column="latest_sales",
    )

    comparison = (
        previous_data
        .merge(
            latest_data,
            on="sales_channel",
            how="outer",
        )
        .fillna(0)
    )

    comparison["sales_change"] = (
        comparison["latest_sales"]
        - comparison["previous_sales"]
    )

    previous_total = float(
        comparison["previous_sales"].sum()
    )

    latest_total = float(
        comparison["latest_sales"].sum()
    )

    total_change = latest_total - previous_total

    total_change_pct = (
        None
        if previous_total == 0
        else total_change / previous_total * 100
    )

    largest_growth = (
        comparison[
            comparison["sales_change"] > 0
        ]
        .sort_values(
            "sales_change",
            ascending=False,
        )
        .head(3)
    )

    largest_declines = (
        comparison[
            comparison["sales_change"] < 0
        ]
        .sort_values(
            "sales_change",
            ascending=True,
        )
        .head(3)
    )

    return {
        "analysis_scope": {
            "data_source": str(DATA_PATH),
            "current_month_excluded": get_current_month_start().strftime("%Y-%m"),
        },
        "month_selection": selection_mode,
        "requested_end_month": end_month,
        "previous_month": previous_month.strftime(
            "%Y-%m"
        ),
        "latest_month": latest_month.strftime(
            "%Y-%m"
        ),
        "comparison_statement": (
            f"Sales changed from "
            f"${previous_total:,.2f} in "
            f"{previous_month.strftime('%Y-%m')} to "
            f"${latest_total:,.2f} in "
            f"{latest_month.strftime('%Y-%m')}."
        ),
        "previous_total_sales": round(
            previous_total,
            2,
        ),
        "latest_total_sales": round(
            latest_total,
            2,
        ),
        "total_change": round(
            total_change,
            2,
        ),
        "total_change_pct": (
            None
            if total_change_pct is None
            else round(
                total_change_pct,
                2,
            )
        ),
        "positive_growth_found": not largest_growth.empty,
        "largest_growth": format_channel_rows(
            largest_growth
        ),
        "largest_declines": format_channel_rows(
            largest_declines
        ),
    }


def analyze_latest_sales_change() -> dict[str, Any]:
    """
    Backward-compatible wrapper.

    Use the latest two completed calendar months.
    """
    return analyze_sales_change(
        end_month=None,
    )


if __name__ == "__main__":
    pprint(
        analyze_latest_sales_change(),
        sort_dicts=False,
    )