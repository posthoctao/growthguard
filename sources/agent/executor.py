from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from sources.agent.planner import (
    AnalysisPlan,
)
from sources.agent.tool_registry import (
    run_single_tool,
)


logger = logging.getLogger(__name__)


ExecutionStatus = Literal[
    "success",
    "partial_success",
    "error",
    "out_of_scope",
]


TOOL_TIMEOUT_SECONDS = int(
    os.getenv(
        "TOOL_TIMEOUT_SECONDS",
        "30",
    )
)


class ExecutionResult(BaseModel):
    """
    Structured output from deterministic tool execution.
    """

    status: ExecutionStatus

    in_scope: bool

    user_goal: str

    selected_tools: list[str] = Field(
        default_factory=list
    )

    successful_tools: list[str] = Field(
        default_factory=list
    )

    failed_tools: list[str] = Field(
        default_factory=list
    )

    needs_rag: bool = False

    rag_status: str = "disabled"

    tool_results: dict[str, Any] = Field(
        default_factory=dict
    )

    execution_notes: list[str] = Field(
        default_factory=list
    )


async def run_tool_with_timeout(
    tool_name: str,
    sales_end_month: str | None,
) -> Any:
    """
    Run one synchronous analytics tool in a worker thread.
    """
    return await asyncio.wait_for(
        asyncio.to_thread(
            run_single_tool,
            tool_name,
            sales_end_month,
        ),
        timeout=TOOL_TIMEOUT_SECONDS,
    )


async def execute_analysis_plan(
    plan: AnalysisPlan,
    sales_end_month: str | None = None,
) -> ExecutionResult:
    """
    Execute a validated analysis plan.
    """
    if not plan.in_scope:
        return ExecutionResult(
            status="out_of_scope",
            in_scope=False,
            user_goal=plan.user_goal,
            needs_rag=False,
            rag_status="disabled",
            execution_notes=[
                "The request is outside the supported analytics scope."
            ],
        )

    selected_tools = [
        step.tool_name
        for step in plan.steps
    ]

    successful_tools: list[str] = []
    failed_tools: list[str] = []
    tool_results: dict[str, Any] = {}
    execution_notes: list[str] = []

    for tool_name in selected_tools:
        try:
            result = await run_tool_with_timeout(
                tool_name=tool_name,
                sales_end_month=sales_end_month,
            )

        except asyncio.TimeoutError:
            failed_tools.append(tool_name)

            execution_notes.append(
                f"{tool_name} timed out."
            )

            logger.exception(
                "Analytics tool timed out: %s",
                tool_name,
            )

        except Exception:
            failed_tools.append(tool_name)

            execution_notes.append(
                f"{tool_name} could not be completed."
            )

            logger.exception(
                "Analytics tool failed: %s",
                tool_name,
            )

        else:
            successful_tools.append(tool_name)
            tool_results[tool_name] = result

    if successful_tools and not failed_tools:
        status: ExecutionStatus = "success"

    elif successful_tools and failed_tools:
        status = "partial_success"

    else:
        status = "error"

    return ExecutionResult(
        status=status,
        in_scope=True,
        user_goal=plan.user_goal,
        selected_tools=selected_tools,
        successful_tools=successful_tools,
        failed_tools=failed_tools,
        needs_rag=False,
        rag_status="disabled",
        tool_results=tool_results,
        execution_notes=execution_notes,
    )