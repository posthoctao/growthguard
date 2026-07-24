from pathlib import Path
from pprint import pprint
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRODUCT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "product_performance_clean.csv"
)

PRODUCT_SALES_CHANNEL_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "product_sales_by_sku_channel_clean.csv"
)

TOP_N = 5

MIN_NET_SALES_FOR_REFUND_PRESSURE = 100_000


def require_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    """Check whether all required columns exist."""

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {dataset_name}: "
            f"{missing_columns}"
        )


def optional_float(value: Any) -> float | None:
    """Convert a value to float while preserving missing values."""

    if value is None or pd.isna(value):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def optional_rounded_float(
    value: Any,
    digits: int = 2,
) -> float | None:
    """Convert a number to rounded float while preserving missing values."""

    number = optional_float(value)

    if number is None:
        return None

    return round(number, digits)


def safe_ratio(
    numerator: Any,
    denominator: Any,
) -> float | None:
    """Calculate a ratio safely."""

    numerator_value = optional_float(numerator)
    denominator_value = optional_float(denominator)

    if numerator_value is None:
        return None

    if denominator_value is None:
        return None

    if denominator_value == 0:
        return None

    return numerator_value / denominator_value


def clean_text_column(
    df: pd.DataFrame,
    column: str,
    fallback_value: str,
) -> pd.DataFrame:
    """Clean one text column."""

    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .fillna(fallback_value)
    )

    return df


def load_product_performance_data() -> pd.DataFrame:
    """
    Load cleaned SKU-level product performance data.

    This file is an all-time SKU-level dataset because the Shopify
    order source does not include a usable order date column.
    """

    df = pd.read_csv(PRODUCT_DATA_PATH)

    required_columns = [
        "sku",
        "product_name",
        "variant_name",
        "product_type",
        "package_type",
        "is_bundle",
        "order_count",
        "unique_order_count",
        "units_sold",
        "net_sales",
        "gross_sales",
        "discounts",
        "refund_amount",
        "quantity_returned",
        "refund_order_count",
        "refund_pressure_ratio",
        "average_selling_price",
        "discount_rate",
    ]

    require_columns(
        df=df,
        required_columns=required_columns,
        dataset_name="product performance data",
    )

    text_columns = {
        "sku": "unknown_sku",
        "product_name": "unknown_product",
        "variant_name": "unknown_variant",
        "product_type": "other",
        "package_type": "unknown",
    }

    for column, fallback_value in text_columns.items():
        df = clean_text_column(
            df=df,
            column=column,
            fallback_value=fallback_value,
        )

    df["is_bundle"] = (
        df["is_bundle"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .astype("boolean")
    )

    numeric_columns = [
        "order_count",
        "unique_order_count",
        "units_sold",
        "net_sales",
        "gross_sales",
        "discounts",
        "refund_amount",
        "quantity_returned",
        "refund_order_count",
        "refund_pressure_ratio",
        "average_selling_price",
        "discount_rate",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.dropna(
            subset=[
                "sku",
                "net_sales",
            ]
        )
        .sort_values(
            "net_sales",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return df


def load_product_sales_channel_data() -> pd.DataFrame:
    """
    Load SKU + sales channel data.

    This file is used only to identify which products perform best
    within major sales channels.
    """

    df = pd.read_csv(
        PRODUCT_SALES_CHANNEL_DATA_PATH
    )

    required_columns = [
        "sku",
        "sales_channel",
        "product_name",
        "variant_name",
        "net_sales",
        "units_sold",
        "order_count",
    ]

    require_columns(
        df=df,
        required_columns=required_columns,
        dataset_name="product sales channel data",
    )

    text_columns = {
        "sku": "unknown_sku",
        "sales_channel": "unknown",
        "product_name": "unknown_product",
        "variant_name": "unknown_variant",
    }

    for column, fallback_value in text_columns.items():
        df = clean_text_column(
            df=df,
            column=column,
            fallback_value=fallback_value,
        )

    numeric_columns = [
        "net_sales",
        "units_sold",
        "order_count",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.dropna(
            subset=[
                "sku",
                "sales_channel",
                "net_sales",
            ]
        )
        .sort_values(
            "net_sales",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return df


def product_rows_to_records(
    product_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Convert product rows into JSON-safe dictionaries.

    Percent fields are converted from decimals to percentages.
    """

    records = []

    for _, row in product_df.iterrows():
        refund_pressure_ratio = optional_float(
            row["refund_pressure_ratio"]
        )

        discount_rate = optional_float(
            row["discount_rate"]
        )

        record = {
            "sku": str(row["sku"]),
            "product_name": str(
                row["product_name"]
            ),
            "variant_name": str(
                row["variant_name"]
            ),
            "product_type": str(
                row["product_type"]
            ),
            "package_type": str(
                row["package_type"]
            ),
            "is_bundle": (
                bool(row["is_bundle"])
                if pd.notna(row["is_bundle"])
                else None
            ),
            "net_sales": optional_rounded_float(
                row["net_sales"],
                2,
            ),
            "gross_sales": optional_rounded_float(
                row["gross_sales"],
                2,
            ),
            "discounts": optional_rounded_float(
                row["discounts"],
                2,
            ),
            "order_count": (
                int(round(float(row["order_count"])))
                if pd.notna(row["order_count"])
                else None
            ),
            "unique_order_count": (
                int(
                    round(
                        float(
                            row["unique_order_count"]
                        )
                    )
                )
                if pd.notna(
                    row["unique_order_count"]
                )
                else None
            ),
            "units_sold": (
                int(round(float(row["units_sold"])))
                if pd.notna(row["units_sold"])
                else None
            ),
            "average_selling_price": (
                optional_rounded_float(
                    row["average_selling_price"],
                    2,
                )
            ),
            "discount_rate_pct": (
                round(discount_rate * 100, 3)
                if discount_rate is not None
                else None
            ),
            "refund_amount": optional_rounded_float(
                row["refund_amount"],
                2,
            ),
            "quantity_returned": (
                int(
                    round(
                        float(
                            row["quantity_returned"]
                        )
                    )
                )
                if pd.notna(
                    row["quantity_returned"]
                )
                else None
            ),
            "refund_order_count": (
                int(
                    round(
                        float(
                            row["refund_order_count"]
                        )
                    )
                )
                if pd.notna(
                    row["refund_order_count"]
                )
                else None
            ),
            "refund_pressure_ratio_pct": (
                round(refund_pressure_ratio * 100, 3)
                if refund_pressure_ratio is not None
                else None
            ),
        }

        records.append(record)

    return records


def build_bundle_vs_single_summary(
    product_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Compare bundle products against single products.

    Unknown package types are excluded because they cannot be
    confidently classified from product naming.
    """

    comparable_df = product_df.loc[
        product_df["package_type"].isin(
            [
                "bundle",
                "single",
            ]
        )
    ].copy()

    if comparable_df.empty:
        return []

    package_summary = (
        comparable_df
        .groupby(
            "package_type",
            as_index=False,
        )
        .agg(
            sku_count=(
                "sku",
                "nunique",
            ),
            product_type_count=(
                "product_type",
                "nunique",
            ),
            net_sales=(
                "net_sales",
                "sum",
            ),
            gross_sales=(
                "gross_sales",
                "sum",
            ),
            discounts=(
                "discounts",
                "sum",
            ),
            units_sold=(
                "units_sold",
                "sum",
            ),
            unique_order_count=(
                "unique_order_count",
                "sum",
            ),
            refund_amount=(
                "refund_amount",
                "sum",
            ),
            quantity_returned=(
                "quantity_returned",
                "sum",
            ),
            refund_order_count=(
                "refund_order_count",
                "sum",
            ),
        )
        .sort_values(
            "net_sales",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    records = []

    for _, row in package_summary.iterrows():
        refund_pressure_ratio = safe_ratio(
            numerator=row["refund_amount"],
            denominator=row["net_sales"],
        )

        discount_rate = safe_ratio(
            numerator=abs(row["discounts"]),
            denominator=row["gross_sales"],
        )

        average_selling_price = safe_ratio(
            numerator=row["net_sales"],
            denominator=row["units_sold"],
        )

        record = {
            "package_type": str(
                row["package_type"]
            ),
            "sku_count": int(row["sku_count"]),
            "product_type_count": int(
                row["product_type_count"]
            ),
            "net_sales": round(
                float(row["net_sales"]),
                2,
            ),
            "units_sold": int(
                round(float(row["units_sold"]))
            ),
            "unique_order_count": int(
                round(
                    float(
                        row["unique_order_count"]
                    )
                )
            ),
            "average_selling_price": (
                round(average_selling_price, 2)
                if average_selling_price is not None
                else None
            ),
            "discount_rate_pct": (
                round(discount_rate * 100, 3)
                if discount_rate is not None
                else None
            ),
            "refund_amount": round(
                float(row["refund_amount"]),
                2,
            ),
            "quantity_returned": int(
                round(
                    float(
                        row["quantity_returned"]
                    )
                )
            ),
            "refund_pressure_ratio_pct": (
                round(
                    refund_pressure_ratio * 100,
                    3,
                )
                if refund_pressure_ratio is not None
                else None
            ),
        }

        records.append(record)

    return records


def build_product_type_summary(
    product_df: pd.DataFrame,
    top_n: int = TOP_N,
) -> list[dict[str, Any]]:
    """Summarize performance by rule-based product family."""

    product_type_summary = (
        product_df
        .groupby(
            "product_type",
            as_index=False,
        )
        .agg(
            sku_count=(
                "sku",
                "nunique",
            ),
            net_sales=(
                "net_sales",
                "sum",
            ),
            units_sold=(
                "units_sold",
                "sum",
            ),
            refund_amount=(
                "refund_amount",
                "sum",
            ),
            discounts=(
                "discounts",
                "sum",
            ),
            gross_sales=(
                "gross_sales",
                "sum",
            ),
        )
        .sort_values(
            "net_sales",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    records = []

    for _, row in product_type_summary.iterrows():
        refund_pressure_ratio = safe_ratio(
            numerator=row["refund_amount"],
            denominator=row["net_sales"],
        )

        discount_rate = safe_ratio(
            numerator=abs(row["discounts"]),
            denominator=row["gross_sales"],
        )

        records.append(
            {
                "product_type": str(
                    row["product_type"]
                ),
                "sku_count": int(row["sku_count"]),
                "net_sales": round(
                    float(row["net_sales"]),
                    2,
                ),
                "units_sold": int(
                    round(float(row["units_sold"]))
                ),
                "refund_amount": round(
                    float(row["refund_amount"]),
                    2,
                ),
                "refund_pressure_ratio_pct": (
                    round(
                        refund_pressure_ratio * 100,
                        3,
                    )
                    if refund_pressure_ratio
                    is not None
                    else None
                ),
                "discount_rate_pct": (
                    round(discount_rate * 100, 3)
                    if discount_rate is not None
                    else None
                ),
            }
        )

    return records


def build_top_products_by_channel(
    product_sales_channel_df: pd.DataFrame,
    top_channel_n: int = 5,
    top_product_n: int = 3,
) -> list[dict[str, Any]]:
    """
    Return top products for the highest-sales channels.

    The analysis is all-time because the source file has no order date.
    """

    channel_summary = (
        product_sales_channel_df
        .groupby(
            "sales_channel",
            as_index=False,
        )
        .agg(
            channel_net_sales=(
                "net_sales",
                "sum",
            ),
            channel_units_sold=(
                "units_sold",
                "sum",
            ),
        )
        .sort_values(
            "channel_net_sales",
            ascending=False,
        )
        .head(top_channel_n)
    )

    records = []

    for _, channel_row in channel_summary.iterrows():
        sales_channel = channel_row["sales_channel"]

        channel_products = (
            product_sales_channel_df.loc[
                product_sales_channel_df[
                    "sales_channel"
                ]
                == sales_channel
            ]
            .sort_values(
                "net_sales",
                ascending=False,
            )
            .head(top_product_n)
        )

        top_products = []

        for _, product_row in channel_products.iterrows():
            top_products.append(
                {
                    "sku": str(product_row["sku"]),
                    "product_name": str(
                        product_row["product_name"]
                    ),
                    "variant_name": str(
                        product_row["variant_name"]
                    ),
                    "net_sales": optional_rounded_float(
                        product_row["net_sales"],
                        2,
                    ),
                    "units_sold": (
                        int(
                            round(
                                float(
                                    product_row[
                                        "units_sold"
                                    ]
                                )
                            )
                        )
                        if pd.notna(
                            product_row["units_sold"]
                        )
                        else None
                    ),
                    "order_count": (
                        int(
                            round(
                                float(
                                    product_row[
                                        "order_count"
                                    ]
                                )
                            )
                        )
                        if pd.notna(
                            product_row["order_count"]
                        )
                        else None
                    ),
                }
            )

        records.append(
            {
                "sales_channel": str(sales_channel),
                "channel_net_sales": round(
                    float(
                        channel_row[
                            "channel_net_sales"
                        ]
                    ),
                    2,
                ),
                "channel_units_sold": int(
                    round(
                        float(
                            channel_row[
                                "channel_units_sold"
                            ]
                        )
                    )
                ),
                "top_products": top_products,
            }
        )

    return records


def get_high_sales_high_refund_pressure_products(
    product_df: pd.DataFrame,
    top_n: int = TOP_N,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Identify products with both meaningful sales volume and elevated
    same-period refund pressure.

    refund_pressure_ratio is a screening metric:
    identified SKU refund amount / identified SKU net sales.

    It is not a strict order-level refund rate.
    """

    eligible_df = product_df.loc[
        (
            product_df["net_sales"]
            >= MIN_NET_SALES_FOR_REFUND_PRESSURE
        )
        & product_df["refund_pressure_ratio"].notna()
        & (
            product_df["refund_amount"]
            > 0
        )
    ].copy()

    if eligible_df.empty:
        return (
            [],
            {
                "minimum_net_sales": (
                    MIN_NET_SALES_FOR_REFUND_PRESSURE
                ),
                "refund_pressure_threshold_pct": None,
                "selection_note": (
                    "No SKU met the minimum sales threshold."
                ),
            },
        )

    refund_pressure_threshold = eligible_df[
        "refund_pressure_ratio"
    ].quantile(0.75)

    high_sales_threshold = eligible_df[
        "net_sales"
    ].quantile(0.75)

    high_risk_df = eligible_df.loc[
        (
            eligible_df["net_sales"]
            >= high_sales_threshold
        )
        & (
            eligible_df["refund_pressure_ratio"]
            >= refund_pressure_threshold
        )
    ].copy()

    if high_risk_df.empty:
        high_risk_df = eligible_df.copy()

    high_risk_df = (
        high_risk_df
        .sort_values(
            [
                "refund_pressure_ratio",
                "refund_amount",
                "net_sales",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return (
        product_rows_to_records(high_risk_df),
        {
            "minimum_net_sales": (
                MIN_NET_SALES_FOR_REFUND_PRESSURE
            ),
            "high_sales_threshold": round(
                float(high_sales_threshold),
                2,
            ),
            "refund_pressure_threshold_pct": round(
                float(refund_pressure_threshold) * 100,
                3,
            ),
            "selection_note": (
                "High-sales, high-refund-pressure SKUs are selected "
                "from products with at least the minimum net sales "
                "threshold, then filtered using the 75th percentile "
                "of both net sales and refund pressure when possible."
            ),
        },
    )


def analyze_product_performance() -> dict[str, Any]:
    """
    Analyze all-time SKU and product performance.

    Includes:
    - top-selling SKUs
    - top SKUs by units sold
    - refund amount and refund pressure
    - high-sales, high-refund-pressure SKUs
    - bundle versus single comparison
    - top products in major sales channels

    Limitation:
    The source order report does not contain a usable order date,
    so this is not a monthly trend analysis.
    """

    product_df = load_product_performance_data()

    product_sales_channel_df = (
        load_product_sales_channel_data()
    )

    top_products_by_net_sales = (
        product_df
        .sort_values(
            "net_sales",
            ascending=False,
        )
        .head(TOP_N)
    )

    top_products_by_units_sold = (
        product_df
        .sort_values(
            [
                "units_sold",
                "net_sales",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(TOP_N)
    )

    top_products_by_refund_amount = (
        product_df
        .loc[
            product_df["refund_amount"] > 0
        ]
        .sort_values(
            [
                "refund_amount",
                "refund_pressure_ratio",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(TOP_N)
    )

    top_products_by_refund_pressure = (
        product_df
        .loc[
            (
                product_df["net_sales"]
                >= MIN_NET_SALES_FOR_REFUND_PRESSURE
            )
            & product_df[
                "refund_pressure_ratio"
            ].notna()
            & (
                product_df["refund_amount"]
                > 0
            )
        ]
        .sort_values(
            [
                "refund_pressure_ratio",
                "refund_amount",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(TOP_N)
    )

    (
        high_sales_high_refund_pressure_products,
        risk_selection_method,
    ) = get_high_sales_high_refund_pressure_products(
        product_df=product_df,
        top_n=TOP_N,
    )

    total_identified_sku_net_sales = (
        product_df["net_sales"]
        .sum()
    )

    total_identified_sku_refund_amount = (
        product_df["refund_amount"]
        .sum()
    )

    overall_refund_pressure_ratio = safe_ratio(
        numerator=total_identified_sku_refund_amount,
        denominator=total_identified_sku_net_sales,
    )

    return {
        "analysis_scope": {
            "analysis_type": (
                "All-time SKU and product performance analysis."
            ),
            "sku_count": int(
                product_df["sku"].nunique()
            ),
            "product_type_count": int(
                product_df["product_type"].nunique()
            ),
            "data_limitations": [
                (
                    "The Shopify order source has no usable order date, "
                    "so product results cannot be compared month over month."
                ),
                (
                    "Product-level sales include only records with an "
                    "identifiable SKU."
                ),
                (
                    "Product-level refunds include only refund records "
                    "with an identifiable SKU."
                ),
                (
                    "Bundle versus single classification is inferred from "
                    "SKU, product name, and variant name rather than an "
                    "original source field."
                ),
                (
                    "refund_pressure_ratio is a risk-screening metric "
                    "based on identified SKU refund amount divided by "
                    "identified SKU net sales. It is not a strict "
                    "order-level refund rate."
                ),
            ],
        },
        "identified_sku_totals": {
            "net_sales": round(
                float(total_identified_sku_net_sales),
                2,
            ),
            "refund_amount": round(
                float(total_identified_sku_refund_amount),
                2,
            ),
            "overall_refund_pressure_ratio_pct": (
                round(
                    overall_refund_pressure_ratio * 100,
                    3,
                )
                if overall_refund_pressure_ratio
                is not None
                else None
            ),
        },
        "top_products_by_net_sales": (
            product_rows_to_records(
                top_products_by_net_sales
            )
        ),
        "top_products_by_units_sold": (
            product_rows_to_records(
                top_products_by_units_sold
            )
        ),
        "top_products_by_refund_amount": (
            product_rows_to_records(
                top_products_by_refund_amount
            )
        ),
        "top_products_by_refund_pressure": (
            product_rows_to_records(
                top_products_by_refund_pressure
            )
        ),
        "high_sales_high_refund_pressure_products": (
            high_sales_high_refund_pressure_products
        ),
        "high_sales_high_refund_pressure_selection": (
            risk_selection_method
        ),
        "bundle_vs_single": (
            build_bundle_vs_single_summary(
                product_df
            )
        ),
        "top_product_types": build_product_type_summary(
            product_df=product_df,
            top_n=TOP_N,
        ),
        "top_products_by_sales_channel": (
            build_top_products_by_channel(
                product_sales_channel_df=
                product_sales_channel_df,
                top_channel_n=5,
                top_product_n=3,
            )
        ),
    }


if __name__ == "__main__":
    result = analyze_product_performance()
    pprint(result, sort_dicts=False)