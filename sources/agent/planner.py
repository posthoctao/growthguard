from __future__ import annotations

import os

from agents import Agent, Runner
from pydantic import BaseModel, Field

from sources.agent.tool_registry import (
    TOOL_DESCRIPTIONS,
    TOOL_ORDER,
    normalize_selected_tools,
)


MODEL_NAME = os.getenv(
    "OPENAI_MODEL",
    "gpt-5-nano",
)


class PlanStep(BaseModel):
    """
    One deterministic analytics tool required for the analysis.
    """

    tool_name: str = Field(
        description="One allowed GrowthGuard analytics tool name."
    )

    reason: str = Field(
        description=(
            "A short explanation of why this tool is needed "
            "for the user's question."
        )
    )


class AnalysisPlan(BaseModel):
    """
    Structured execution plan created before running analytics tools.
    """

    in_scope: bool = Field(
        description=(
            "Whether the question is within GrowthGuard DTC "
            "growth analytics scope."
        )
    )

    user_goal: str = Field(
        description=(
            "A concise description of the user's analytical goal."
        )
    )

    steps: list[PlanStep] = Field(
        description=(
            "An ordered list of deterministic analytics tools "
            "required to answer the question."
        )
    )

    needs_rag: bool = Field(
        description=(
            "Whether internal documents, metric definitions, product "
            "documentation, campaign notes, or qualitative context "
            "would materially improve the answer."
        )
    )

    planner_note: str = Field(
        description=(
            "A concise internal note describing the analysis approach."
        )
    )


def build_tool_catalog() -> str:
    """
    Build a formatted catalog of available deterministic tools.
    """
    return "\n".join(
        (
            f"- {tool_name}: {TOOL_DESCRIPTIONS[tool_name]}"
        )
        for tool_name in TOOL_ORDER
    )


def build_planner_agent() -> Agent:
    """
    Create the Planner Agent.

    The Planner selects tools and creates a structured plan.
    It does not execute tools or generate the final user response.
    """
    tool_catalog = build_tool_catalog()

    return Agent(
        name="GrowthGuard Analysis Planner",
        model=MODEL_NAME,
        output_type=AnalysisPlan,
        instructions=f"""
You are the planning component of the GrowthGuard DTC Growth
Analytics Agent.

Your responsibility is to convert a user question into a concise,
structured analytics plan.

You must not calculate metrics.
You must not call analytics tools.
You must not write the final user-facing answer.

Available deterministic analytics tools:

{tool_catalog}

Planning rules:

1. Select only tool names listed in the available tool catalog.
2. Select the minimum number of tools needed to answer the question.
3. Do not include duplicate tools.
4. Keep the plan focused and practical.
5. Use "sales" for revenue, sales performance, monthly sales changes,
   sales channels, channel growth, or channel declines.
6. Use "funnel" for sessions, bounce rate, cart additions, checkout,
   conversion rate, or website funnel performance.
7. Use "refund" for refund amount, refund orders, returned quantity,
   refund changes, or recently refunded SKUs.
8. Use "subscription" for active subscribers, new subscribers, churn,
   cancellation, deactivation, or net subscriber changes.
9. Use "cohort" for cohort retention, Month 1, Month 2, Month 3,
   early lifecycle retention, or subscriber drop-off.
10. Use "marketing" for campaigns, flows, email, SMS, open rate,
    click rate, placed-order performance, or marketing performance.
11. Use "product" for SKU performance, product sales, refund pressure,
    bundle versus single comparison, or product channel performance.
12. When comparing subscription health with retention, select both
    "subscription" and "cohort".
13. When comparing product refund risk with subscription retention,
    select "product", "subscription", and "cohort".
14. Set needs_rag=true only when internal documentation, definitions,
    policies, campaign briefs, customer feedback, product descriptions,
    or other qualitative context is required.
15. Set needs_rag=false when structured analytics tools are enough.
16. Set in_scope=false only when the question is unrelated to
    GrowthGuard DTC growth analytics.
17. When in_scope=false, return an empty steps list.
18. Do not invent metrics, tool results, data sources, or conclusions.
""".strip(),
    )


def contains_any_keyword(
    text: str,
    keywords: list[str],
) -> bool:
    """
    Check whether a normalized user question contains any keyword.
    """
    return any(
        keyword in text
        for keyword in keywords
    )


def add_required_tools(
    question: str,
    selected_tools: list[str],
) -> list[str]:
    """
    Apply deterministic safety rules for known multi-domain questions.

    These rules reduce the risk of the Planner omitting a necessary tool.
    """
    normalized_question = question.lower()

    required_tools = selected_tools.copy()

    subscription_keywords = [
        "subscription",
        "subscriber",
        "churn",
        "deactivation",
        "cancellation",
        "active subscriber",
        "订阅",
        "订阅用户",
        "活跃订阅",
        "停用",
        "退订",
        "流失",
    ]

    retention_keywords = [
        "retention",
        "cohort",
        "month 1",
        "month 2",
        "month 3",
        "m1",
        "m2",
        "m3",
        "留存",
        "队列",
    ]

    product_keywords = [
        "product",
        "sku",
        "bundle",
        "single",
        "产品",
        "商品",
        "套装",
        "单品",
    ]

    refund_keywords = [
        "refund",
        "return",
        "退款",
        "退货",
    ]

    has_subscription_topic = contains_any_keyword(
        normalized_question,
        subscription_keywords,
    )

    has_retention_topic = contains_any_keyword(
        normalized_question,
        retention_keywords,
    )

    has_product_topic = contains_any_keyword(
        normalized_question,
        product_keywords,
    )

    has_refund_topic = contains_any_keyword(
        normalized_question,
        refund_keywords,
    )

    if has_subscription_topic:
        required_tools.append("subscription")

    if has_retention_topic:
        required_tools.append("cohort")

    if has_product_topic and has_refund_topic:
        required_tools.append("product")

    return normalize_selected_tools(
        required_tools
    )


def validate_analysis_plan(
    question: str,
    raw_plan: AnalysisPlan,
) -> AnalysisPlan:
    """
    Validate and normalize Planner output before execution.

    This removes invalid tool names, removes duplicates, and applies
    deterministic multi-domain tool requirements.
    """
    if not raw_plan.in_scope:
        return AnalysisPlan(
            in_scope=False,
            user_goal=raw_plan.user_goal.strip(),
            steps=[],
            needs_rag=False,
            planner_note=(
                "The question is outside the supported GrowthGuard "
                "DTC growth analytics scope."
            ),
        )

    requested_tools = [
        step.tool_name
        for step in raw_plan.steps
    ]

    selected_tools = normalize_selected_tools(
        requested_tools
    )

    selected_tools = add_required_tools(
        question=question,
        selected_tools=selected_tools,
    )

    if not selected_tools:
        return AnalysisPlan(
            in_scope=False,
            user_goal=raw_plan.user_goal.strip(),
            steps=[],
            needs_rag=False,
            planner_note=(
                "No valid GrowthGuard analytics tool was selected."
            ),
        )

    reason_by_tool = {
        step.tool_name.strip().lower(): step.reason.strip()
        for step in raw_plan.steps
        if isinstance(step.tool_name, str)
        and isinstance(step.reason, str)
        and step.reason.strip()
    }

    validated_steps = [
        PlanStep(
            tool_name=tool_name,
            reason=reason_by_tool.get(
                tool_name,
                TOOL_DESCRIPTIONS[tool_name],
            ),
        )
        for tool_name in selected_tools
    ]

    return AnalysisPlan(
        in_scope=True,
        user_goal=raw_plan.user_goal.strip(),
        steps=validated_steps,
        needs_rag=raw_plan.needs_rag,
        planner_note=raw_plan.planner_note.strip(),
    )


async def create_analysis_plan(
    question: str,
) -> AnalysisPlan:
    """
    Create a validated analytics plan for a GrowthGuard user question.

    This function only plans the work. It does not execute tools.
    """
    normalized_question = " ".join(
        question.strip().split()
    )

    if not normalized_question:
        raise ValueError(
            "Question cannot be empty."
        )

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. "
            "Please configure it before running the Planner."
        )

    planner_agent = build_planner_agent()

    result = await Runner.run(
        planner_agent,
        normalized_question,
        max_turns=1,
    )

    raw_plan = result.final_output

    if not isinstance(raw_plan, AnalysisPlan):
        raise RuntimeError(
            "Planner did not return a valid AnalysisPlan."
        )

    return validate_analysis_plan(
        question=normalized_question,
        raw_plan=raw_plan,
    )