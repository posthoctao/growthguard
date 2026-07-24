from __future__ import annotations

import json
import os
import re
from typing import Any

from agents import Agent, Runner

from sources.agent.executor import ExecutionResult
from sources.agent.planner import AnalysisPlan


MODEL_NAME = os.getenv(
    "OPENAI_MODEL",
    "gpt-5-nano",
)


def contains_chinese(text: str) -> bool:
    """
    Return True when text contains Chinese characters.
    """
    return bool(
        re.search(
            r"[\u4e00-\u9fff]",
            text,
        )
    )


def build_out_of_scope_message(
    question: str,
) -> str:
    """
    Return a user-facing message for unsupported questions.
    """
    if contains_chinese(question):
        return (
            "这个问题目前不属于 GrowthGuard 增长分析助手的支持范围。"
            "你可以咨询销售、订阅、留存、退款、网站漏斗、"
            "产品或营销表现等问题。"
        )

    return (
        "This question is outside the current scope of the "
        "GrowthGuard Growth Analytics Assistant. "
        "You can ask about sales, subscriptions, retention, refunds, "
        "website funnel performance, products, or marketing."
    )


def build_data_unavailable_message(
    question: str,
) -> str:
    """
    Return a user-facing message when no tool result is available.
    """
    if contains_chinese(question):
        return (
            "暂时无法完成这次分析，因为当前可用的数据结果不足。"
            "请稍后重试，或换一种更具体的业务问题表述。"
        )

    return (
        "This analysis could not be completed because the available "
        "data results were insufficient. Please try again later or "
        "ask a more specific business question."
    )


def build_evidence_package(
    question: str,
    plan: AnalysisPlan,
    execution: ExecutionResult,
) -> dict[str, Any]:
    """
    Build the evidence package provided to the Final Response Agent.

    Only deterministic tool results are passed as metric evidence.
    The model must not create facts that are absent from this package.
    """
    successful_results = {
        tool_name: tool_output
        for tool_name, tool_output in execution.tool_results.items()
        if tool_output.get("status") == "success"
    }

    failed_results = {
        tool_name: {
            "error_type": tool_output.get("error_type"),
            "message": tool_output.get("message"),
        }
        for tool_name, tool_output in execution.tool_results.items()
        if tool_output.get("status") == "error"
    }

    return {
        "user_question": question,
        "analysis_goal": plan.user_goal,
        "selected_tools": execution.selected_tools,
        "successful_tools": execution.successful_tools,
        "failed_tools": execution.failed_tools,
        "needs_rag": execution.needs_rag,
        "rag_status": execution.rag_status,
        "execution_notes": execution.execution_notes,
        "deterministic_evidence": successful_results,
        "failed_tool_details": failed_results,
    }


def build_final_response_agent() -> Agent:
    """
    Create the final user-facing response agent.

    This agent does not call tools. It only converts verified deterministic
    evidence into a concise business answer.
    """
    return Agent(
        name="GrowthGuard Final Response Agent",
        model=MODEL_NAME,
        instructions="""
You are the final response component of the GrowthGuard DTC Growth
Analytics Agent.

Your task is to write a clear, concise, user-facing business answer
using only the deterministic evidence package provided in the input.

You do not call tools.
You do not create an analysis plan.
You do not invent metrics, explanations, trends, rankings, or causes.

Language rules:
1. Respond in the same primary language used in the user's question.
2. If the user writes mainly in Chinese, answer in Chinese.
3. If the user writes mainly in English, answer in English.
4. If the user mixes languages, use the dominant language.
5. If the language is unclear, answer in English.

Evidence rules:
1. The deterministic_evidence section is the only source of truth
   for business metrics and analytical conclusions.
2. Never invent values, percentages, dates, channel shares, rankings,
   customer behavior, causal explanations, or business facts.
3. Never use information from failed tools as evidence.
4. Do not claim that one metric caused another unless the evidence
   explicitly proves causation.
5. Preserve all time ranges, analysis scopes, and limitations stated
   in the deterministic evidence.
6. If product analysis is labeled all-time, clearly preserve that scope.
7. If marketing results are estimated, clearly preserve that limitation.
8. If Flow data is described as a snapshot, do not describe it as
   a month-over-month trend.
9. If only two adjacent months are compared, use month-over-month
   language and do not describe the result as a long-term trend.
10. Never call a month-over-month comparison year-over-year.
11. Do not expose internal terms such as Planner, Executor, tool,
    function, JSON, API, RAG, model, or system prompt.

Formatting rules:
1. Start with a direct answer or concise conclusion.
2. Then provide only the most important supporting evidence.
3. Use short sections or bullets only when they improve readability.
4. Format money clearly.
5. Format percentages with a percent sign when evidence provides
   percentage values.
6. If evidence provides a decimal rate, convert it to percentage form
   only when the evidence clearly defines it as a rate.
7. When discussing a rate change, use percentage points when appropriate.
8. Keep the response business-focused and easy to understand.
9. Do not add generic consulting frameworks unless the user explicitly
   asks for recommendations or next steps.

RAG handling rules:
1. If needs_rag is false, answer only from deterministic evidence.
2. If needs_rag is true and rag_status is pending_connection, do not
   fabricate qualitative context from internal documents.
3. In that case, you may state that the available data supports the
   quantitative conclusion, while the underlying qualitative reasons
   require further validation.

Partial-result rules:
1. If some tools failed, answer only with successful evidence.
2. Briefly state that the available data supports only a partial view
   when the missing result materially affects the user's question.
3. Do not mention technical error messages or internal tool failures.

Return only the final user-facing answer.
""".strip(),
    )


async def generate_final_response(
    question: str,
    plan: AnalysisPlan,
    execution: ExecutionResult,
) -> str:
    """
    Generate the final user-facing response from a completed execution.

    Args:
        question:
            Original user question.

        plan:
            Validated analysis plan returned by the Planner.

        execution:
            Standardized deterministic execution result returned by
            the Executor.

    Returns:
        Final response in the user's primary language.
    """
    normalized_question = " ".join(
        question.strip().split()
    )

    if not normalized_question:
        raise ValueError(
            "Question cannot be empty."
        )

    if not execution.in_scope:
        return build_out_of_scope_message(
            normalized_question
        )

    if not execution.successful_tools:
        return build_data_unavailable_message(
            normalized_question
        )

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. "
            "Please configure it before generating a response."
        )

    evidence_package = build_evidence_package(
        question=normalized_question,
        plan=plan,
        execution=execution,
    )

    input_text = (
        "Generate the final user-facing answer using this evidence package.\n\n"
        + json.dumps(
            evidence_package,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    response_agent = build_final_response_agent()

    result = await Runner.run(
        response_agent,
        input_text,
        max_turns=1,
    )

    answer = str(
        result.final_output
    ).strip()

    if not answer:
        return build_data_unavailable_message(
            normalized_question
        )

    return answer