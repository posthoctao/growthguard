from __future__ import annotations

from sources.agent.planner import (
    AnalysisPlan,
    PlanStep,
)
from sources.agent.tool_registry import (
    TOOL_ORDER,
)
from sources.core.exceptions import (
    PlanningError,
)


MAX_TOOLS_PER_REQUEST = 4

ALLOWED_TOOLS = set(TOOL_ORDER)


def validate_analysis_plan(
    plan: AnalysisPlan,
) -> AnalysisPlan:
    """
    Validate and normalize a planner-generated analysis plan.
    """
    if not isinstance(plan, AnalysisPlan):
        raise PlanningError(
            "Planner returned an invalid output type."
        )

    if not plan.in_scope:
        return plan.model_copy(
            update={
                "steps": [],
                "needs_rag": False,
            }
        )

    normalized_steps: list[PlanStep] = []
    seen_tools: set[str] = set()

    for step in plan.steps:
        tool_name = step.tool_name.strip().lower()

        if tool_name not in ALLOWED_TOOLS:
            raise PlanningError(
                f"Planner selected an unsupported tool: {tool_name}"
            )

        if tool_name in seen_tools:
            continue

        seen_tools.add(tool_name)

        normalized_steps.append(
            PlanStep(
                tool_name=tool_name,
                reason=step.reason.strip(),
            )
        )

    if not normalized_steps:
        raise PlanningError(
            "An in-scope request did not select any tools."
        )

    if len(normalized_steps) > MAX_TOOLS_PER_REQUEST:
        raise PlanningError(
            (
                "Planner selected too many tools. "
                f"Maximum allowed: {MAX_TOOLS_PER_REQUEST}."
            )
        )

    return plan.model_copy(
        update={
            "steps": normalized_steps,
            "needs_rag": False,
        }
    )