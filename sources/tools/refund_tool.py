from pathlib import Path
from pprint import pprint

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFUND_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "refunds_clean.csv"
)


MISSING_SKU_VALUES = {
    "",
    "unknown",
    "nan",
    "none",
    "null",
    "n/a",
    "na",
    "not_provided",
}


def load_refund_data() -> pd.DataFrame:
    """Load and validate cleaned refund data."""

    if not REFUND_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Refund data file not found: {REFUND_DATA_PATH}"
        )

    df = pd.read_csv(REFUND_DATA_PATH)

    required_columns = {
        "month",
        "order_id",
        "refund_amount",
        "quantity_returned",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Refund data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df["month"] = pd.to_datetime(
        df["month"],
        errors="coerce",
    )

    df["refund_amount"] = pd.to_numeric(
        df["refund_amount"],
        errors="coerce",
    )

    df["quantity_returned"] = pd.to_numeric(
        df["quantity_returned"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "month",
            "order_id",
            "refund_amount",
            "quantity_returned",
        ]
    ).copy()

    df["order_id"] = df["order_id"].astype(str).str.strip()

    if "product_variant_sku" not in df.columns:
        df["product_variant_sku"] = ""

    df["product_variant_sku"] = (
        df["product_variant_sku"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return df


def get_latest_complete_months(
    df: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return the latest two complete calendar months.

    The current calendar month is excluded because it may be incomplete.
    """

    current_month = pd.Timestamp.today().to_period("M")

    complete_months = (
        df.loc[
            df["month"].dt.to_period("M") < current_month,
            "month",
        ]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if len(complete_months) < 2:
        raise ValueError(
            "At least two complete months of refund data are required."
        )

    previous_month = complete_months[-2]
    latest_month = complete_months[-1]

    return previous_month, latest_month


def summarize_refunds_by_month(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create monthly refund summaries."""

    monthly_summary = (
        df.groupby("month", as_index=False)
        .agg(
            refund_amount=("refund_amount", "sum"),
            refund_order_count=("order_id", "nunique"),
            quantity_returned=("quantity_returned", "sum"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )

    return monthly_summary


def get_top_refunded_products(
    df: pd.DataFrame,
    month: pd.Timestamp,
    top_n: int = 5,
) -> list[dict]:
    """
    Return top refunded SKUs for one month.

    Missing or placeholder SKU values are excluded rather than
    being reported as a real product named 'unknown'.
    """

    month_df = df.loc[df["month"] == month].copy()

    if month_df.empty:
        return []

    sku_lower = (
        month_df["product_variant_sku"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    known_sku_df = month_df.loc[
        ~sku_lower.isin(MISSING_SKU_VALUES)
    ].copy()

    if known_sku_df.empty:
        return []

    product_summary = (
        known_sku_df.groupby(
            "product_variant_sku",
            as_index=False,
        )
        .agg(
            refund_amount=("refund_amount", "sum"),
            quantity_returned=("quantity_returned", "sum"),
            refund_order_count=("order_id", "nunique"),
        )
        .sort_values(
            by=[
                "refund_amount",
                "quantity_returned",
            ],
            ascending=False,
        )
        .head(top_n)
    )

    return [
        {
            "product_variant_sku": row[
                "product_variant_sku"
            ],
            "refund_amount": round(
                float(row["refund_amount"]),
                2,
            ),
            "quantity_returned": int(
                row["quantity_returned"]
            ),
            "refund_order_count": int(
                row["refund_order_count"]
            ),
        }
        for _, row in product_summary.iterrows()
    ]


def get_sku_data_quality(
    df: pd.DataFrame,
    month: pd.Timestamp,
) -> dict:
    """
    Measure whether refund rows have usable SKU information.

    This prevents the agent from treating missing SKU values as real products.
    """

    month_df = df.loc[df["month"] == month].copy()

    if month_df.empty:
        return {
            "total_refund_rows": 0,
            "rows_with_identified_sku": 0,
            "rows_with_missing_sku": 0,
            "identified_sku_refund_amount": 0.0,
            "missing_sku_refund_amount": 0.0,
            "sku_data_is_sufficient": False,
        }

    sku_lower = (
        month_df["product_variant_sku"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    missing_sku_mask = sku_lower.isin(MISSING_SKU_VALUES)

    identified_sku_df = month_df.loc[
        ~missing_sku_mask
    ].copy()

    missing_sku_df = month_df.loc[
        missing_sku_mask
    ].copy()

    return {
        "total_refund_rows": int(len(month_df)),
        "rows_with_identified_sku": int(
            len(identified_sku_df)
        ),
        "rows_with_missing_sku": int(
            len(missing_sku_df)
        ),
        "identified_sku_refund_amount": round(
            float(
                identified_sku_df["refund_amount"].sum()
            ),
            2,
        ),
        "missing_sku_refund_amount": round(
            float(
                missing_sku_df["refund_amount"].sum()
            ),
            2,
        ),
        "sku_data_is_sufficient": bool(
            len(identified_sku_df) > 0
        ),
    }


def judge_refund_status(
    current_value: float,
    previous_value: float,
) -> str:
    """
    Judge whether a refund metric improved or worsened.

    For refund metrics:
    - higher = worsened
    - lower = improved
    """

    if current_value > previous_value:
        return "worsened"

    if current_value < previous_value:
        return "improved"

    return "stable"


def calculate_change(
    current_value: float,
    previous_value: float,
) -> tuple[float, float | None]:
    """Calculate absolute and percentage change."""

    absolute_change = current_value - previous_value

    if previous_value == 0:
        return absolute_change, None

    percentage_change = (
        absolute_change / previous_value
    ) * 100

    return absolute_change, percentage_change


def analyze_latest_refund_change() -> dict:
    """
    Compare the latest two complete months of refund performance.
    """

    refund_df = load_refund_data()

    previous_month, latest_month = get_latest_complete_months(
        refund_df
    )

    monthly_summary = summarize_refunds_by_month(refund_df)

    previous_row = monthly_summary.loc[
        monthly_summary["month"] == previous_month
    ].iloc[0]

    latest_row = monthly_summary.loc[
        monthly_summary["month"] == latest_month
    ].iloc[0]

    previous_refund_amount = float(
        previous_row["refund_amount"]
    )
    latest_refund_amount = float(
        latest_row["refund_amount"]
    )

    previous_refund_orders = int(
        previous_row["refund_order_count"]
    )
    latest_refund_orders = int(
        latest_row["refund_order_count"]
    )

    previous_quantity_returned = float(
        previous_row["quantity_returned"]
    )
    latest_quantity_returned = float(
        latest_row["quantity_returned"]
    )

    refund_amount_change, refund_amount_change_pct = (
        calculate_change(
            latest_refund_amount,
            previous_refund_amount,
        )
    )

    refund_order_change, refund_order_change_pct = (
        calculate_change(
            latest_refund_orders,
            previous_refund_orders,
        )
    )

    quantity_returned_change, quantity_returned_change_pct = (
        calculate_change(
            latest_quantity_returned,
            previous_quantity_returned,
        )
    )

    top_refunded_products = get_top_refunded_products(
        refund_df,
        latest_month,
    )

    sku_data_quality = get_sku_data_quality(
        refund_df,
        latest_month,
    )

    if not sku_data_quality["sku_data_is_sufficient"]:
        product_note = (
            "The available refund data does not identify "
            "refunded products or SKUs clearly."
        )
    else:
        product_note = (
            "Top refunded products are reported only for rows "
            "with an identified SKU."
        )

    return {
        "previous_month": previous_month.strftime("%Y-%m"),
        "latest_month": latest_month.strftime("%Y-%m"),
        "previous_refund_amount": round(
            previous_refund_amount,
            2,
        ),
        "latest_refund_amount": round(
            latest_refund_amount,
            2,
        ),
        "refund_amount_change": round(
            refund_amount_change,
            2,
        ),
        "refund_amount_change_pct": (
            round(refund_amount_change_pct, 2)
            if refund_amount_change_pct is not None
            else None
        ),
        "refund_amount_status": judge_refund_status(
            latest_refund_amount,
            previous_refund_amount,
        ),
        "previous_refund_orders": previous_refund_orders,
        "latest_refund_orders": latest_refund_orders,
        "refund_order_change": refund_order_change,
        "refund_order_change_pct": (
            round(refund_order_change_pct, 2)
            if refund_order_change_pct is not None
            else None
        ),
        "refund_order_status": judge_refund_status(
            latest_refund_orders,
            previous_refund_orders,
        ),
        "previous_quantity_returned": round(
            previous_quantity_returned,
            2,
        ),
        "latest_quantity_returned": round(
            latest_quantity_returned,
            2,
        ),
        "quantity_returned_change": round(
            quantity_returned_change,
            2,
        ),
        "quantity_returned_change_pct": (
            round(quantity_returned_change_pct, 2)
            if quantity_returned_change_pct is not None
            else None
        ),
        "quantity_returned_status": judge_refund_status(
            latest_quantity_returned,
            previous_quantity_returned,
        ),
        "top_refunded_products": top_refunded_products,
        "sku_data_quality": sku_data_quality,
        "product_data_note": product_note,
        "summary": (
            f"Refund amount changed from "
            f"${previous_refund_amount:,.2f} in "
            f"{previous_month.strftime('%Y-%m')} to "
            f"${latest_refund_amount:,.2f} in "
            f"{latest_month.strftime('%Y-%m')}."
        ),
    }


if __name__ == "__main__":
    result = analyze_latest_refund_change()
    pprint(result, sort_dicts=False)