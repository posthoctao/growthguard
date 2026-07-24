from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sources.tools.cohort_tool import (
    analyze_early_cohort_retention,
)
from sources.tools.funnel_tool import (
    analyze_latest_funnel_change,
)
from sources.tools.marketing_tool import (
    analyze_marketing_performance,
)
from sources.tools.product_tool import (
    analyze_product_performance,
)
from sources.tools.refund_tool import (
    analyze_latest_refund_change,
)
from sources.tools.sales_tool import (
    analyze_sales_change,
)
from sources.tools.subscription_tool import (
    analyze_latest_subscription_change,
)


# ============================================================
# Tool configuration
# ============================================================

TOOL_ORDER = [
    "sales",
    "funnel",
    "refund",
    "subscription",
    "cohort",
    "marketing",
    "product",
]


TOOL_DESCRIPTIONS = {
    "sales": (
        "Monthly sales performance, sales changes, channel growth, "
        "and channel decline."
    ),
    "funnel": (
        "Website sessions, bounce rate, cart addition, checkout, "
        "and conversion funnel changes."
    ),
    "refund": (
        "Latest monthly refund amount, refund orders, returned quantity, "
        "and top refunded SKUs."
    ),
    "subscription": (
        "Latest subscription health, active subscribers, new subscribers, "
        "churn, deactivation, and net subscriber change."
    ),
    "cohort": (
        "Early lifecycle subscriber cohort retention, Month 1 to Month 3 "
        "retention, and early drop-off."
    ),
    "marketing": (
        "Email Campaign monthly performance and current live Flow snapshot."
    ),
    "product": (
        "All-time SKU and product performance, product refund pressure, "
        "bundle versus single comparison, and top products by channel."
    ),
}


# All tools except sales have no input parameters in the current version.
STANDARD_TOOL_FUNCTIONS: dict[
    str,
    Callable[[], dict[str, Any]],
] = {
    "funnel": analyze_latest_funnel_change,
    "refund": analyze_latest_refund_change,
    "subscription": analyze_latest_subscription_change,
    "cohort": analyze_early_cohort_retention,
    "marketing": analyze_marketing_performance,
    "product": analyze_product_performance,
}


# ============================================================
# Validation helpers
# ============================================================

def get_available_tool_names() -> list[str]:
    """
    Return all supported deterministic tool names in execution order.
    """
    return TOOL_ORDER.copy()


def get_tool_descriptions() -> dict[str, str]:
    """
    Return descriptions for all available tools.

    This will later be passed into the Planner prompt so the Planner
    knows what each tool can and cannot analyze.
    """
    return TOOL_DESCRIPTIONS.copy()


def normalize_selected_tools(
    selected_tools: list[str],
) -> list[str]:
    """
    Validate, deduplicate, and order requested tool names.

    Invalid names are ignored instead of raising an error so one invalid
    Planner output does not stop valid tools from running.
    """
    if not isinstance(selected_tools, list):
        raise ValueError(
            "selected_tools must be a list of tool names."
        )

    normalized_tools: list[str] = []

    for tool_name in selected_tools:
        if not isinstance(tool_name, str):
            continue

        cleaned_name = tool_name.strip().lower()

        if cleaned_name not in TOOL_ORDER:
            continue

        if cleaned_name not in normalized_tools:
            normalized_tools.append(cleaned_name)

    return [
        tool_name
        for tool_name in TOOL_ORDER
        if tool_name in normalized_tools
    ]


# ============================================================
# Tool execution
# ============================================================

def run_single_tool(
    tool_name: str,
    sales_end_month: str | None = None,
) -> dict[str, Any]:
    """
    Run one deterministic GrowthGuard analytics tool.

    Args:
        tool_name:
            One valid tool name from TOOL_ORDER.

        sales_end_month:
            Optional YYYY-MM end month for the sales tool only.

            Example:
            sales_end_month="2026-05"
            compares 2026-04 with 2026-05.

            When None, the sales tool automatically uses the latest
            complete months after excluding configured partial months.

    Returns:
        A dictionary containing either:
        - status="success" with analytics result
        - status="error" with a readable error message
    """
    if tool_name not in TOOL_ORDER:
        return {
            "status": "error",
            "tool_name": tool_name,
            "error_type": "unknown_tool",
            "message": (
                f"Unsupported tool: {tool_name}. "
                f"Available tools: {', '.join(TOOL_ORDER)}"
            ),
        }

    try:
        if tool_name == "sales":
            result = analyze_sales_change(
                end_month=sales_end_month,
            )
        else:
            tool_function = STANDARD_TOOL_FUNCTIONS[tool_name]
            result = tool_function()

        return {
            "status": "success",
            "tool_name": tool_name,
            "tool_description": TOOL_DESCRIPTIONS[tool_name],
            "result": result,
        }

    except Exception as error:
        return {
            "status": "error",
            "tool_name": tool_name,
            "tool_description": TOOL_DESCRIPTIONS[tool_name],
            "error_type": type(error).__name__,
            "message": str(error),
        }


def execute_tools(
    selected_tools: list[str],
    sales_end_month: str | None = None,
) -> dict[str, Any]:
    """
    Run multiple deterministic tools selected by a Planner.

    Args:
        selected_tools:
            Tool names selected by a Planner or rule-based router.

        sales_end_month:
            Optional target month used only when sales is selected.

    Returns:
        A standard execution package that later components can use:
        - Planner
        - Executor
        - Final Answer Agent
        - Evaluation tests
    """
    normalized_tools = normalize_selected_tools(
        selected_tools
    )

    if not normalized_tools:
        return {
            "status": "error",
            "selected_tools": [],
            "tool_results": {},
            "message": (
                "No valid analytics tools were selected."
            ),
        }

    tool_results: dict[str, dict[str, Any]] = {}

    for tool_name in normalized_tools:
        tool_results[tool_name] = run_single_tool(
            tool_name=tool_name,
            sales_end_month=sales_end_month,
        )

    successful_tools = [
        tool_name
        for tool_name, tool_output in tool_results.items()
        if tool_output.get("status") == "success"
    ]

    failed_tools = [
        tool_name
        for tool_name, tool_output in tool_results.items()
        if tool_output.get("status") == "error"
    ]

    overall_status = (
        "success"
        if not failed_tools
        else "partial_success"
        if successful_tools
        else "error"
    )

    return {
        "status": overall_status,
        "selected_tools": normalized_tools,
        "sales_end_month": sales_end_month,
        "successful_tools": successful_tools,
        "failed_tools": failed_tools,
        "tool_results": tool_results,
    }
