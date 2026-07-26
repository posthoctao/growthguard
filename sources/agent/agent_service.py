from __future__ import annotations

import logging
import os
import re
from typing import Literal

from pydantic import BaseModel

from sources.agent.context_resolver import (
    resolve_user_question,
)
from sources.agent.executor import (
    ExecutionResult,
    execute_analysis_plan,
)
from sources.agent.final_response_agent import (
    generate_final_response,
)
from sources.agent.memory_extractor import (
    extract_long_term_memory_changes,
)
from sources.agent.planner import (
    AnalysisPlan,
    create_analysis_plan,
)
from sources.core.exceptions import (
    AgentApplicationError,
    PlanningError,
    ToolExecutionError,
    UpstreamServiceError,
)
from sources.guardrails.input_validator import (
    normalize_question,
)
from sources.guardrails.output_validator import (
    validate_final_answer,
)
from sources.guardrails.plan_validator import (
    validate_analysis_plan,
)
from sources.memory.long_term_memory import (
    apply_memory_changes,
    get_user_memory_context,
    normalize_user_id,
)
from sources.memory.session_manager import (
    get_conversation_context,
    normalize_session_id,
    save_conversation_turn,
)
from sources.observability.logger import (
    hash_identifier,
    log_event,
)
from sources.observability.request_tracker import (
    RequestTracker,
)


RunStatus = Literal[
    "success",
    "partial_success",
    "error",
    "out_of_scope",
]


MONTH_NAME_TO_NUMBER = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


class AgentRunResult(BaseModel):
    """
    Standard result returned by the GrowthGuard workflow.
    """

    status: RunStatus

    answer: str

    in_scope: bool

    session_id: str | None = None

    user_id: str | None = None

    original_question: str

    resolved_question: str

    selected_tools: list[str]

    successful_tools: list[str]

    failed_tools: list[str]

    sales_end_month: str | None = None

    plan: AnalysisPlan

    execution: ExecutionResult


def format_year_month(
    year_text: str,
    month_number: int | str,
) -> str:
    """
    Normalize a year and month into YYYY-MM.
    """
    year = int(year_text)
    month = int(month_number)

    if not 1 <= month <= 12:
        raise ValueError(
            f"Invalid month number: {month}"
        )

    return f"{year:04d}-{month:02d}"


def extract_requested_sales_end_month(
    question: str,
) -> str | None:
    """
    Extract an explicitly requested sales month.
    """
    numeric_patterns = [
        re.compile(
            r"(?<!\d)(20\d{2})\s*[-/.]\s*"
            r"(0?[1-9]|1[0-2])(?!\d)"
        ),
        re.compile(
            r"(?<!\d)(20\d{2})\s*年\s*"
            r"(0?[1-9]|1[0-2])\s*月"
        ),
    ]

    for pattern in numeric_patterns:
        match = pattern.search(question)

        if match:
            year_text, month_text = (
                match.groups()
            )

            return format_year_month(
                year_text=year_text,
                month_number=month_text,
            )

    month_name_pattern = re.compile(
        r"\b("
        r"january|february|march|april|may|june|"
        r"july|august|september|october|november|december"
        r")\s*,?\s*(20\d{2})\b",
        re.IGNORECASE,
    )

    match = month_name_pattern.search(
        question
    )

    if match:
        month_name, year_text = (
            match.groups()
        )

        return format_year_month(
            year_text=year_text,
            month_number=(
                MONTH_NAME_TO_NUMBER[
                    month_name.lower()
                ]
            ),
        )

    year_month_name_pattern = re.compile(
        r"\b(20\d{2})\s+("
        r"january|february|march|april|may|june|"
        r"july|august|september|october|november|december"
        r")\b",
        re.IGNORECASE,
    )

    match = year_month_name_pattern.search(
        question
    )

    if match:
        year_text, month_name = (
            match.groups()
        )

        return format_year_month(
            year_text=year_text,
            month_number=(
                MONTH_NAME_TO_NUMBER[
                    month_name.lower()
                ]
            ),
        )

    return None


def get_overall_status(
    plan: AnalysisPlan,
    execution: ExecutionResult,
) -> RunStatus:
    """
    Convert workflow state into one overall status.
    """
    if not plan.in_scope:
        return "out_of_scope"

    if execution.status == "success":
        return "success"

    if execution.status == "partial_success":
        return "partial_success"

    return "error"


async def load_conversation_context(
    session_id: str,
    request_id: str,
) -> str:
    """
    Load short-term session history without failing the request.
    """
    try:
        return await get_conversation_context(
            session_id=session_id,
        )

    except Exception as error:
        log_event(
            "session_memory_load_failed",
            level=logging.WARNING,
            request_id=request_id,
            session_hash=hash_identifier(
                session_id
            ),
            error_type=type(error).__name__,
        )

        return ""


async def load_user_memory_safely(
    user_id: str,
    request_id: str,
) -> str:
    """
    Load long-term user memory without failing the request.
    """
    try:
        return get_user_memory_context(
            user_id=user_id,
        )

    except Exception as error:
        log_event(
            "long_term_memory_load_failed",
            level=logging.WARNING,
            request_id=request_id,
            user_hash=hash_identifier(
                user_id
            ),
            error_type=type(error).__name__,
        )

        return ""


async def resolve_question_safely(
    question: str,
    conversation_context: str,
    user_memory_context: str,
    request_id: str,
) -> str:
    """
    Resolve a follow-up question using session and user memory.
    """
    if (
        not conversation_context
        and not user_memory_context
    ):
        return question

    try:
        resolved_question = (
            await resolve_user_question(
                question=question,
                conversation_context=(
                    conversation_context
                ),
                user_memory_context=(
                    user_memory_context
                ),
            )
        )

        return normalize_question(
            resolved_question
        )

    except Exception as error:
        log_event(
            "context_resolution_failed",
            level=logging.WARNING,
            request_id=request_id,
            error_type=type(error).__name__,
        )

        return question


async def create_plan_safely(
    question: str,
) -> AnalysisPlan:
    """
    Create and validate a planner output.
    """
    try:
        raw_plan = await create_analysis_plan(
            question
        )

        return validate_analysis_plan(
            raw_plan
        )

    except PlanningError:
        raise

    except Exception as error:
        raise PlanningError(
            "Planner execution failed."
        ) from error


def build_final_question(
    question: str,
    user_memory_context: str,
) -> str:
    """
    Attach durable response preferences to the final-answer request.
    """
    if not user_memory_context.strip():
        return question

    return f"""
User question:

{question}

Durable user preferences:

{user_memory_context}

Apply these preferences silently.

Rules:
- Do not mention that long-term memory was used.
- Do not treat stored preferences as business evidence.
- Do not use memory as a source for current KPI values.
- Current metrics must come only from deterministic analytics tools.
""".strip()


async def generate_answer_safely(
    question: str,
    plan: AnalysisPlan,
    execution: ExecutionResult,
) -> str:
    """
    Generate and validate the final response.
    """
    try:
        answer = await generate_final_response(
            question=question,
            plan=plan,
            execution=execution,
        )

    except Exception as error:
        raise UpstreamServiceError(
            "Final response generation failed."
        ) from error

    return validate_final_answer(
        answer
    )


async def save_memory_safely(
    session_id: str,
    user_message: str,
    assistant_message: str,
    request_id: str,
) -> None:
    """
    Save short-term session memory without failing the answer.
    """
    try:
        await save_conversation_turn(
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
        )

    except Exception as error:
        log_event(
            "session_memory_save_failed",
            level=logging.WARNING,
            request_id=request_id,
            session_hash=hash_identifier(
                session_id
            ),
            error_type=type(error).__name__,
        )


async def update_long_term_memory_safely(
    user_id: str,
    session_id: str | None,
    user_message: str,
    existing_memory_context: str,
    request_id: str,
) -> None:
    """
    Extract and save durable user preferences.
    """
    try:
        changes = (
            await extract_long_term_memory_changes(
                user_message=user_message,
                existing_memory_context=(
                    existing_memory_context
                ),
            )
        )

        if not changes:
            return

        applied_count = apply_memory_changes(
            user_id=user_id,
            changes=changes,
            source_session_id=session_id,
        )

        log_event(
            "long_term_memory_updated",
            request_id=request_id,
            user_hash=hash_identifier(
                user_id
            ),
            applied_count=applied_count,
        )

    except Exception as error:
        log_event(
            "long_term_memory_save_failed",
            level=logging.WARNING,
            request_id=request_id,
            user_hash=hash_identifier(
                user_id
            ),
            error_type=type(error).__name__,
        )


async def run_agent_with_details(
    question: str,
    session_id: str | None = None,
    user_id: str | None = None,
    request_id: str | None = None,
) -> AgentRunResult:
    """
    Run the complete guarded, observable and memory-enabled workflow.
    """
    original_question = normalize_question(
        question
    )

    tracker = RequestTracker.create(
        request_id=request_id,
        session_id=session_id,
        question_length=len(
            original_question
        ),
    )

    selected_tools: list[str] = []
    successful_tools: list[str] = []
    failed_tools: list[str] = []

    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise UpstreamServiceError(
                "OPENAI_API_KEY is not configured."
            )

        normalized_session_id: str | None = (
            None
        )

        normalized_user_id: str | None = (
            None
        )

        conversation_context = ""
        user_memory_context = ""

        if session_id is not None:
            normalized_session_id = (
                normalize_session_id(
                    session_id
                )
            )

            with tracker.stage(
                "memory_load"
            ):
                conversation_context = (
                    await load_conversation_context(
                        session_id=(
                            normalized_session_id
                        ),
                        request_id=(
                            tracker.request_id
                        ),
                    )
                )

        if user_id is not None:
            normalized_user_id = (
                normalize_user_id(
                    user_id
                )
            )

            with tracker.stage(
                "long_term_memory_load"
            ):
                user_memory_context = (
                    await load_user_memory_safely(
                        user_id=(
                            normalized_user_id
                        ),
                        request_id=(
                            tracker.request_id
                        ),
                    )
                )

        with tracker.stage(
            "context_resolution"
        ):
            resolved_question = (
                await resolve_question_safely(
                    question=(
                        original_question
                    ),
                    conversation_context=(
                        conversation_context
                    ),
                    user_memory_context=(
                        user_memory_context
                    ),
                    request_id=(
                        tracker.request_id
                    ),
                )
            )

        tracker.context_resolved = (
            resolved_question
            != original_question
        )

        with tracker.stage(
            "planning"
        ):
            plan = await create_plan_safely(
                resolved_question
            )

        selected_tools = [
            step.tool_name
            for step in plan.steps
        ]

        sales_end_month = None

        if "sales" in selected_tools:
            sales_end_month = (
                extract_requested_sales_end_month(
                    resolved_question
                )
            )

        with tracker.stage(
            "execution"
        ):
            execution = (
                await execute_analysis_plan(
                    plan=plan,
                    sales_end_month=(
                        sales_end_month
                    ),
                )
            )

        successful_tools = list(
            execution.successful_tools
        )

        failed_tools = list(
            execution.failed_tools
        )

        if (
            plan.in_scope
            and execution.status == "error"
        ):
            raise ToolExecutionError(
                "All selected analytics tools failed."
            )

        final_question = build_final_question(
            question=resolved_question,
            user_memory_context=(
                user_memory_context
            ),
        )

        with tracker.stage(
            "final_response"
        ):
            answer = (
                await generate_answer_safely(
                    question=final_question,
                    plan=plan,
                    execution=execution,
                )
            )

        if normalized_session_id:
            with tracker.stage(
                "memory_save"
            ):
                await save_memory_safely(
                    session_id=(
                        normalized_session_id
                    ),
                    user_message=(
                        original_question
                    ),
                    assistant_message=answer,
                    request_id=(
                        tracker.request_id
                    ),
                )

        if normalized_user_id:
            with tracker.stage(
                "long_term_memory_save"
            ):
                await update_long_term_memory_safely(
                    user_id=(
                        normalized_user_id
                    ),
                    session_id=(
                        normalized_session_id
                    ),
                    user_message=(
                        original_question
                    ),
                    existing_memory_context=(
                        user_memory_context
                    ),
                    request_id=(
                        tracker.request_id
                    ),
                )

        overall_status = get_overall_status(
            plan=plan,
            execution=execution,
        )

        tracker.complete(
            status=overall_status,
            in_scope=plan.in_scope,
            selected_tools=selected_tools,
            successful_tools=(
                successful_tools
            ),
            failed_tools=failed_tools,
        )

        return AgentRunResult(
            status=overall_status,
            answer=answer,
            in_scope=plan.in_scope,
            session_id=(
                normalized_session_id
            ),
            user_id=(
                normalized_user_id
            ),
            original_question=(
                original_question
            ),
            resolved_question=(
                resolved_question
            ),
            selected_tools=(
                selected_tools
            ),
            successful_tools=(
                successful_tools
            ),
            failed_tools=failed_tools,
            sales_end_month=(
                sales_end_month
            ),
            plan=plan,
            execution=execution,
        )

    except AgentApplicationError as error:
        tracker.fail(
            error_code=error.error_code,
            error=error,
            selected_tools=selected_tools,
            successful_tools=(
                successful_tools
            ),
            failed_tools=failed_tools,
        )

        raise

    except Exception as error:
        tracker.fail(
            error_code="UNEXPECTED_ERROR",
            error=error,
            selected_tools=selected_tools,
            successful_tools=(
                successful_tools
            ),
            failed_tools=failed_tools,
        )

        raise


async def run_agent(
    question: str,
    session_id: str | None = None,
    user_id: str | None = None,
    request_id: str | None = None,
) -> str:
    """
    Run the workflow and return only the final answer.
    """
    result = await run_agent_with_details(
        question=question,
        session_id=session_id,
        user_id=user_id,
        request_id=request_id,
    )

    return result.answer