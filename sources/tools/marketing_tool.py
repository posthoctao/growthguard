from pathlib import Path
from pprint import pprint
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAMPAIGN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "campaign_performance_clean.csv"
)

FLOW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "flow_performance_clean.csv"
)

MIN_RECIPIENTS_FOR_RATE_RANKING = 500
RATE_STATUS_TOLERANCE = 0.0005


RATE_OUTCOME_MAPPING = {
    "open_rate": "estimated_opens",
    "click_rate": "estimated_clicks",
    "unsubscribe_rate": "estimated_unsubscribes",
    "bounce_rate": "estimated_bounces",
    "placed_order_rate": "estimated_placed_orders",
    "loop_subscription_started_rate": (
        "estimated_loop_subscription_starts"
    ),
}


RATE_DIRECTION = {
    "open_rate": "higher_is_better",
    "click_rate": "higher_is_better",
    "unsubscribe_rate": "lower_is_better",
    "bounce_rate": "lower_is_better",
    "placed_order_rate": "higher_is_better",
    "loop_subscription_started_rate": "higher_is_better",
}


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


def to_boolean_series(series: pd.Series) -> pd.Series:
    """Convert mixed True/False text values into a boolean Series."""

    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def optional_float(value: Any) -> float | None:
    """Convert a value to float, while preserving missing values."""

    if value is None or pd.isna(value):
        return None

    return float(value)


def optional_rounded_float(
    value: Any,
    digits: int = 2,
) -> float | None:
    """Convert a value to rounded float, while preserving missing values."""

    number = optional_float(value)

    if number is None:
        return None

    return round(number, digits)


def safe_rate(
    numerator: Any,
    denominator: Any,
) -> float | None:
    """Calculate a rate safely."""

    numerator_value = optional_float(numerator)
    denominator_value = optional_float(denominator)

    if numerator_value is None or denominator_value is None:
        return None

    if denominator_value == 0:
        return None

    return numerator_value / denominator_value


def clean_rate_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert rate columns to numeric and remove impossible values."""

    for rate_column in RATE_OUTCOME_MAPPING:
        df[rate_column] = pd.to_numeric(
            df[rate_column],
            errors="coerce",
        )

        invalid_rate_mask = (
            df[rate_column].notna()
            & (
                (df[rate_column] < 0)
                | (df[rate_column] > 1)
            )
        )

        df.loc[
            invalid_rate_mask,
            rate_column,
        ] = pd.NA

    return df


def ensure_estimated_outcome_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure estimated outcome columns exist.

    Estimated outcomes are calculated as:
    total_recipients × reported rate
    """

    for rate_column, outcome_column in RATE_OUTCOME_MAPPING.items():
        if outcome_column not in df.columns:
            df[outcome_column] = (
                df["total_recipients"]
                * df[rate_column]
            )

        df[outcome_column] = pd.to_numeric(
            df[outcome_column],
            errors="coerce",
        )

    return df


def load_campaign_data() -> pd.DataFrame:
    """
    Load cleaned Email campaign data.

    Campaign data is time-series data, so it can be analyzed
    month-over-month.
    """

    df = pd.read_csv(CAMPAIGN_DATA_PATH)

    required_columns = [
        "campaign_message_id",
        "campaign_message_name",
        "list_segment",
        "send_date",
        "campaign_month",
        "total_recipients",
        "open_rate",
        "click_rate",
        "unsubscribe_rate",
        "bounce_rate",
        "placed_order_rate",
        "loop_subscription_started_rate",
        "is_complete_month",
    ]

    require_columns(
        df=df,
        required_columns=required_columns,
        dataset_name="campaign data",
    )

    df["send_date"] = pd.to_datetime(
        df["send_date"],
        errors="coerce",
    )

    df["campaign_month"] = pd.to_datetime(
        df["campaign_month"],
        errors="coerce",
    )

    text_columns = [
        "campaign_message_id",
        "campaign_message_name",
        "list_segment",
    ]

    for column in text_columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    df["campaign_message_name"] = (
        df["campaign_message_name"]
        .fillna("unknown_campaign")
    )

    df["list_segment"] = (
        df["list_segment"]
        .fillna("unknown_segment")
    )

    df["total_recipients"] = pd.to_numeric(
        df["total_recipients"],
        errors="coerce",
    )

    df = clean_rate_columns(df)
    df = ensure_estimated_outcome_columns(df)

    df["is_complete_month"] = to_boolean_series(
        df["is_complete_month"]
    )

    df = (
        df.dropna(
            subset=[
                "campaign_message_id",
                "send_date",
                "campaign_month",
                "total_recipients",
            ]
        )
        .sort_values(
            [
                "send_date",
                "campaign_message_id",
            ]
        )
        .reset_index(drop=True)
    )

    return df


def load_flow_data() -> pd.DataFrame:
    """
    Load cleaned Klaviyo Flow data.

    Flow data is a current performance snapshot, not a monthly time series.
    """

    df = pd.read_csv(FLOW_DATA_PATH)

    required_columns = [
        "flow_id",
        "flow_name",
        "message_channel",
        "status",
        "is_live_flow",
        "total_recipients",
        "open_rate",
        "click_rate",
        "unsubscribe_rate",
        "bounce_rate",
        "placed_order_rate",
        "loop_subscription_started_rate",
    ]

    require_columns(
        df=df,
        required_columns=required_columns,
        dataset_name="flow data",
    )

    text_columns = [
        "flow_id",
        "flow_name",
        "message_channel",
        "status",
    ]

    for column in text_columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    df["flow_name"] = (
        df["flow_name"]
        .fillna("unknown_flow")
    )

    df["message_channel"] = (
        df["message_channel"]
        .str.lower()
        .fillna("unknown")
    )

    df["status"] = (
        df["status"]
        .str.lower()
        .fillna("unknown")
    )

    df["is_live_flow"] = to_boolean_series(
        df["is_live_flow"]
    )

    df["total_recipients"] = pd.to_numeric(
        df["total_recipients"],
        errors="coerce",
    )

    df = clean_rate_columns(df)
    df = ensure_estimated_outcome_columns(df)

    df = (
        df.dropna(
            subset=[
                "flow_id",
                "flow_name",
                "message_channel",
                "status",
                "total_recipients",
            ]
        )
        .sort_values(
            [
                "is_live_flow",
                "total_recipients",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return df


def add_weighted_rates(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate recipient-weighted rates.

    Example:
    weighted click rate =
    total estimated clicks / total recipients
    """

    recipient_base = summary_df[
        "total_recipients"
    ].replace(0, pd.NA)

    for rate_column, outcome_column in RATE_OUTCOME_MAPPING.items():
        summary_df[rate_column] = (
            summary_df[outcome_column]
            / recipient_base
        )

    return summary_df


def build_monthly_campaign_summary(
    campaign_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build monthly campaign performance using complete months only.

    One campaign can have multiple List/Segment rows, so all monthly
    rates are calculated with recipient-weighted aggregation.
    """

    complete_campaign_df = campaign_df.loc[
        campaign_df["is_complete_month"]
    ].copy()

    if complete_campaign_df.empty:
        raise ValueError(
            "No complete campaign months are available."
        )

    aggregation_rules = {
        "total_recipients": (
            "total_recipients",
            "sum",
        ),
        "campaign_segment_rows": (
            "campaign_message_id",
            "size",
        ),
        "unique_campaign_messages": (
            "campaign_message_id",
            "nunique",
        ),
    }

    for outcome_column in RATE_OUTCOME_MAPPING.values():
        aggregation_rules[outcome_column] = (
            outcome_column,
            lambda values: values.sum(min_count=1),
        )

    monthly_summary = (
        complete_campaign_df
        .groupby(
            "campaign_month",
            as_index=False,
        )
        .agg(**aggregation_rules)
        .sort_values("campaign_month")
        .reset_index(drop=True)
    )

    monthly_summary = add_weighted_rates(
        monthly_summary
    )

    return monthly_summary


def get_latest_two_complete_campaign_months(
    monthly_campaign_summary: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the latest two complete campaign months."""

    available_months = sorted(
        monthly_campaign_summary["campaign_month"]
        .dropna()
        .unique()
    )

    if len(available_months) < 2:
        raise ValueError(
            "Not enough complete campaign months to compare."
        )

    previous_month = pd.Timestamp(available_months[-2])
    latest_month = pd.Timestamp(available_months[-1])

    return previous_month, latest_month


def judge_rate_status(
    metric_name: str,
    previous_value: float,
    latest_value: float,
) -> str:
    """Classify a metric as improved, worsened, or stable."""

    change = latest_value - previous_value

    if abs(change) <= RATE_STATUS_TOLERANCE:
        return "stable"

    metric_direction = RATE_DIRECTION[metric_name]

    if metric_direction == "higher_is_better":
        return "improved" if change > 0 else "worsened"

    return "improved" if change < 0 else "worsened"


def build_campaign_metric_changes(
    previous_row: pd.Series,
    latest_row: pd.Series,
) -> list[dict[str, Any]]:
    """Build month-over-month changes for key campaign metrics."""

    metric_changes = []

    for metric_name in RATE_DIRECTION:
        previous_value = optional_float(
            previous_row[metric_name]
        )

        latest_value = optional_float(
            latest_row[metric_name]
        )

        if previous_value is None or latest_value is None:
            continue

        metric_changes.append(
            {
                "metric": metric_name,
                "previous_value": round(
                    previous_value,
                    6,
                ),
                "latest_value": round(
                    latest_value,
                    6,
                ),
                "change_percentage_points": round(
                    (latest_value - previous_value) * 100,
                    3,
                ),
                "status": judge_rate_status(
                    metric_name=metric_name,
                    previous_value=previous_value,
                    latest_value=latest_value,
                ),
            }
        )

    return metric_changes


def judge_campaign_status(
    metric_changes: list[dict[str, Any]],
) -> str:
    """Judge overall campaign performance direction."""

    improved_count = sum(
        item["status"] == "improved"
        for item in metric_changes
    )

    worsened_count = sum(
        item["status"] == "worsened"
        for item in metric_changes
    )

    if improved_count > worsened_count:
        return "improved"

    if worsened_count > improved_count:
        return "worsened"

    return "mixed"


def build_campaign_message_summary(
    campaign_df: pd.DataFrame,
    target_month: pd.Timestamp,
) -> pd.DataFrame:
    """
    Aggregate campaign rows into campaign-message-level performance.

    A campaign message can have multiple segment rows, so they are
    combined before ranking top campaigns.
    """

    month_df = campaign_df.loc[
        campaign_df["campaign_month"] == target_month
    ].copy()

    if month_df.empty:
        raise ValueError(
            f"No campaign data found for {target_month:%Y-%m}."
        )

    aggregation_rules = {
        "send_date": (
            "send_date",
            "min",
        ),
        "total_recipients": (
            "total_recipients",
            "sum",
        ),
        "list_segment_count": (
            "list_segment",
            "nunique",
        ),
    }

    for outcome_column in RATE_OUTCOME_MAPPING.values():
        aggregation_rules[outcome_column] = (
            outcome_column,
            lambda values: values.sum(min_count=1),
        )

    campaign_summary = (
        month_df
        .groupby(
            [
                "campaign_message_id",
                "campaign_message_name",
            ],
            as_index=False,
        )
        .agg(**aggregation_rules)
    )

    campaign_summary = add_weighted_rates(
        campaign_summary
    )

    return (
        campaign_summary
        .sort_values(
            [
                "estimated_placed_orders",
                "placed_order_rate",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )


def build_live_flow_summary(
    flow_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate currently live flows.

    A single Flow may have multiple rows, especially across Email and SMS,
    so the aggregation key includes Flow ID, name, and channel.
    """

    live_flow_df = flow_df.loc[
        flow_df["is_live_flow"]
    ].copy()

    if live_flow_df.empty:
        raise ValueError(
            "No live flows are available for analysis."
        )

    aggregation_rules = {
        "total_recipients": (
            "total_recipients",
            "sum",
        ),
        "flow_report_rows": (
            "flow_id",
            "size",
        ),
    }

    for outcome_column in RATE_OUTCOME_MAPPING.values():
        aggregation_rules[outcome_column] = (
            outcome_column,
            lambda values: values.sum(min_count=1),
        )

    live_flow_summary = (
        live_flow_df
        .groupby(
            [
                "flow_id",
                "flow_name",
                "message_channel",
            ],
            as_index=False,
        )
        .agg(**aggregation_rules)
    )

    live_flow_summary = add_weighted_rates(
        live_flow_summary
    )

    return (
        live_flow_summary
        .sort_values(
            [
                "estimated_placed_orders",
                "placed_order_rate",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )


def monthly_campaign_row_to_record(
    row: pd.Series,
) -> dict[str, Any]:
    """Convert one monthly campaign row into a JSON-safe dictionary."""

    record = {
        "total_recipients": int(
            round(float(row["total_recipients"]))
        ),
        "campaign_segment_rows": int(
            row["campaign_segment_rows"]
        ),
        "unique_campaign_messages": int(
            row["unique_campaign_messages"]
        ),
        "estimated_placed_orders": optional_rounded_float(
            row["estimated_placed_orders"],
            2,
        ),
        "estimated_loop_subscription_starts": (
            optional_rounded_float(
                row[
                    "estimated_loop_subscription_starts"
                ],
                2,
            )
        ),
    }

    for rate_column in RATE_OUTCOME_MAPPING:
        rate_value = optional_float(
            row[rate_column]
        )

        record[f"{rate_column}_pct"] = (
            round(rate_value * 100, 3)
            if rate_value is not None
            else None
        )

    return record


def campaign_rows_to_records(
    campaign_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Convert campaign DataFrame rows into JSON-safe dictionaries."""

    records = []

    for _, row in campaign_df.iterrows():
        record = {
            "campaign_message_id": str(
                row["campaign_message_id"]
            ),
            "campaign_message_name": str(
                row["campaign_message_name"]
            ),
            "send_date": pd.Timestamp(
                row["send_date"]
            ).strftime("%Y-%m-%d"),
            "total_recipients": int(
                round(float(row["total_recipients"]))
            ),
            "list_segment_count": int(
                row["list_segment_count"]
            ),
            "estimated_placed_orders": optional_rounded_float(
                row["estimated_placed_orders"],
                2,
            ),
        }

        for rate_column in RATE_OUTCOME_MAPPING:
            rate_value = optional_float(
                row[rate_column]
            )

            record[f"{rate_column}_pct"] = (
                round(rate_value * 100, 3)
                if rate_value is not None
                else None
            )

        records.append(record)

    return records


def flow_rows_to_records(
    flow_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Convert flow DataFrame rows into JSON-safe dictionaries."""

    records = []

    for _, row in flow_df.iterrows():
        record = {
            "flow_id": str(row["flow_id"]),
            "flow_name": str(row["flow_name"]),
            "message_channel": str(
                row["message_channel"]
            ),
            "total_recipients": int(
                round(float(row["total_recipients"]))
            ),
            "flow_report_rows": int(
                row["flow_report_rows"]
            ),
            "estimated_placed_orders": optional_rounded_float(
                row["estimated_placed_orders"],
                2,
            ),
            "estimated_loop_subscription_starts": (
                optional_rounded_float(
                    row[
                        "estimated_loop_subscription_starts"
                    ],
                    2,
                )
            ),
        }

        for rate_column in RATE_OUTCOME_MAPPING:
            rate_value = optional_float(
                row[rate_column]
            )

            record[f"{rate_column}_pct"] = (
                round(rate_value * 100, 3)
                if rate_value is not None
                else None
            )

        records.append(record)

    return records


def build_flow_channel_summary(
    live_flow_summary: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Build Email and SMS summaries separately.

    Rates are recipient-weighted.
    """

    aggregation_rules = {
        "total_recipients": (
            "total_recipients",
            "sum",
        ),
        "unique_live_flow_ids": (
            "flow_id",
            "nunique",
        ),
        "live_flow_report_rows": (
            "flow_id",
            "size",
        ),
    }

    for outcome_column in RATE_OUTCOME_MAPPING.values():
        aggregation_rules[outcome_column] = (
            outcome_column,
            lambda values: values.sum(min_count=1),
        )

    channel_summary_df = (
        live_flow_summary
        .groupby(
            "message_channel",
            as_index=False,
        )
        .agg(**aggregation_rules)
        .sort_values("message_channel")
        .reset_index(drop=True)
    )

    channel_summary_df = add_weighted_rates(
        channel_summary_df
    )

    records = []

    for _, row in channel_summary_df.iterrows():
        record = {
            "message_channel": str(
                row["message_channel"]
            ),
            "unique_live_flow_ids": int(
                row["unique_live_flow_ids"]
            ),
            "live_flow_report_rows": int(
                row["live_flow_report_rows"]
            ),
            "total_recipients": int(
                round(float(row["total_recipients"]))
            ),
            "estimated_placed_orders": optional_rounded_float(
                row["estimated_placed_orders"],
                2,
            ),
            "estimated_loop_subscription_starts": (
                optional_rounded_float(
                    row[
                        "estimated_loop_subscription_starts"
                    ],
                    2,
                )
            ),
        }

        for rate_column in RATE_OUTCOME_MAPPING:
            rate_value = optional_float(
                row[rate_column]
            )

            record[f"{rate_column}_pct"] = (
                round(rate_value * 100, 3)
                if rate_value is not None
                else None
            )

        records.append(record)

    return records


def get_top_campaigns(
    campaign_summary: pd.DataFrame,
    top_n: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """
    Return top campaigns by:
    1. estimated placed orders
    2. placed order rate
    """

    top_by_estimated_orders = (
        campaign_summary
        .sort_values(
            [
                "estimated_placed_orders",
                "placed_order_rate",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(top_n)
    )

    eligible_for_rate_ranking = campaign_summary.loc[
        (
            campaign_summary["total_recipients"]
            >= MIN_RECIPIENTS_FOR_RATE_RANKING
        )
        & campaign_summary["placed_order_rate"].notna()
    ].copy()

    if eligible_for_rate_ranking.empty:
        eligible_for_rate_ranking = (
            campaign_summary
            .loc[
                campaign_summary[
                    "placed_order_rate"
                ].notna()
            ]
            .copy()
        )

    top_by_placed_order_rate = (
        eligible_for_rate_ranking
        .sort_values(
            [
                "placed_order_rate",
                "estimated_placed_orders",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(top_n)
    )

    return {
        "top_by_estimated_placed_orders": (
            campaign_rows_to_records(
                top_by_estimated_orders
            )
        ),
        "top_by_placed_order_rate": (
            campaign_rows_to_records(
                top_by_placed_order_rate
            )
        ),
    }


def get_top_live_flows(
    live_flow_summary: pd.DataFrame,
    top_n: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """
    Return top live flows by:
    1. estimated placed orders
    2. placed order rate
    """

    top_by_estimated_orders = (
        live_flow_summary
        .sort_values(
            [
                "estimated_placed_orders",
                "placed_order_rate",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(top_n)
    )

    eligible_for_rate_ranking = live_flow_summary.loc[
        (
            live_flow_summary["total_recipients"]
            >= MIN_RECIPIENTS_FOR_RATE_RANKING
        )
        & live_flow_summary["placed_order_rate"].notna()
    ].copy()

    if eligible_for_rate_ranking.empty:
        eligible_for_rate_ranking = (
            live_flow_summary
            .loc[
                live_flow_summary[
                    "placed_order_rate"
                ].notna()
            ]
            .copy()
        )

    top_by_placed_order_rate = (
        eligible_for_rate_ranking
        .sort_values(
            [
                "placed_order_rate",
                "estimated_placed_orders",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(top_n)
    )

    return {
        "top_by_estimated_placed_orders": (
            flow_rows_to_records(
                top_by_estimated_orders
            )
        ),
        "top_by_placed_order_rate": (
            flow_rows_to_records(
                top_by_placed_order_rate
            )
        ),
    }


def analyze_campaign_performance() -> dict[str, Any]:
    """
    Analyze the latest two complete months of Email campaign performance.
    """

    campaign_df = load_campaign_data()

    monthly_campaign_summary = build_monthly_campaign_summary(
        campaign_df
    )

    previous_month, latest_month = (
        get_latest_two_complete_campaign_months(
            monthly_campaign_summary
        )
    )

    previous_row = monthly_campaign_summary.loc[
        monthly_campaign_summary["campaign_month"]
        == previous_month
    ].iloc[0]

    latest_row = monthly_campaign_summary.loc[
        monthly_campaign_summary["campaign_month"]
        == latest_month
    ].iloc[0]

    metric_changes = build_campaign_metric_changes(
        previous_row=previous_row,
        latest_row=latest_row,
    )

    latest_campaign_summary = (
        build_campaign_message_summary(
            campaign_df=campaign_df,
            target_month=latest_month,
        )
    )

    previous_estimated_orders = optional_float(
        previous_row["estimated_placed_orders"]
    )

    latest_estimated_orders = optional_float(
        latest_row["estimated_placed_orders"]
    )

    estimated_orders_change = None
    estimated_orders_change_pct = None

    if (
        previous_estimated_orders is not None
        and latest_estimated_orders is not None
    ):
        estimated_orders_change = round(
            latest_estimated_orders
            - previous_estimated_orders,
            2,
        )

        if previous_estimated_orders != 0:
            estimated_orders_change_pct = round(
                (
                    estimated_orders_change
                    / previous_estimated_orders
                )
                * 100,
                2,
            )

    return {
        "previous_month": previous_month.strftime("%Y-%m"),
        "latest_month": latest_month.strftime("%Y-%m"),
        "campaign_status": judge_campaign_status(
            metric_changes
        ),
        "aggregation_note": (
            "Campaign rates are recipient-weighted across "
            "campaign-segment report rows. Estimated outcome counts "
            "are calculated from total recipients multiplied by "
            "the reported rate."
        ),
        "previous_month_performance": (
            monthly_campaign_row_to_record(
                previous_row
            )
        ),
        "latest_month_performance": (
            monthly_campaign_row_to_record(
                latest_row
            )
        ),
        "estimated_placed_orders_change": (
            estimated_orders_change
        ),
        "estimated_placed_orders_change_pct": (
            estimated_orders_change_pct
        ),
        "metric_changes": metric_changes,
        "top_campaigns_latest_month": get_top_campaigns(
            latest_campaign_summary,
            top_n=5,
        ),
    }


def analyze_live_flow_performance() -> dict[str, Any]:
    """
    Analyze current live Flow performance.

    The Flow source file is a point-in-time snapshot. It does not include
    monthly timestamps, so this function does not calculate month-over-month
    Flow changes.
    """

    flow_df = load_flow_data()

    live_flow_summary = build_live_flow_summary(
        flow_df
    )

    return {
        "analysis_scope": {
            "total_flow_report_rows": int(
                len(flow_df)
            ),
            "live_flow_report_rows": int(
                flow_df["is_live_flow"].sum()
            ),
            "unique_live_flow_ids": int(
                live_flow_summary["flow_id"].nunique()
            ),
            "analysis_type": (
                "Current live Flow performance snapshot."
            ),
        },
        "aggregation_note": (
            "Flow rates are recipient-weighted. Estimated outcome "
            "counts are calculated from total recipients multiplied "
            "by the reported rate."
        ),
        "channel_summary": build_flow_channel_summary(
            live_flow_summary
        ),
        "top_live_flows": get_top_live_flows(
            live_flow_summary,
            top_n=5,
        ),
    }


def analyze_marketing_performance() -> dict[str, Any]:
    """
    Combine Campaign and Flow marketing analysis.

    Campaign:
    - latest two complete months
    - monthly trend comparison

    Flow:
    - current live performance snapshot
    - no monthly trend because source data has no date field
    """

    campaign_result = analyze_campaign_performance()

    flow_result = analyze_live_flow_performance()

    return {
        "marketing_scope_note": (
            "Campaign data is analyzed as a monthly trend. "
            "Flow data is analyzed as a current live-performance "
            "snapshot because the Flow report does not include "
            "send dates or monthly timestamps."
        ),
        "campaign_result": campaign_result,
        "flow_result": flow_result,
    }


if __name__ == "__main__":
    result = analyze_marketing_performance()
    pprint(result, sort_dicts=False)