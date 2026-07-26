from __future__ import annotations

import os

from agents import Agent, Runner
from pydantic import BaseModel, Field


MODEL_NAME = os.getenv(
    "OPENAI_MODEL",
    "gpt-5-nano",
)


class ResolvedQuestion(BaseModel):
    """
    A standalone question resolved from available context.
    """

    standalone_question: str = Field(
        description=(
            "A complete standalone version of the current "
            "user question."
        )
    )


def build_context_resolver_agent() -> Agent:
    """
    Create the conversation-context resolver.
    """
    return Agent(
        name="GrowthGuard Context Resolver",
        model=MODEL_NAME,
        output_type=ResolvedQuestion,
        instructions="""
Rewrite the latest message as a complete standalone GrowthGuard
analytics question.

You may use:
- recent messages from the current session;
- durable long-term user preferences.

Rules:
1. Resolve references such as "that", "it", "compared with that",
   "那", "这个", "上面", "相比呢" and "按我平时关注的方向".
2. Preserve dates, products, channels, metrics and comparisons.
3. Use long-term memory only for durable preferences, recurring focus,
   team role and response requirements.
4. Never treat a value stored in memory as a current KPI.
5. Current business values must still be obtained from deterministic
   analytics tools.
6. Do not calculate metrics.
7. Do not answer the question.
8. Do not invent missing business context.
9. Write in the same primary language as the latest message.
10. When the latest message is already standalone, make only minimal
    wording changes.
""".strip(),
    )


async def resolve_user_question(
    question: str,
    conversation_context: str = "",
    user_memory_context: str = "",
) -> str:
    """
    Resolve one message using session and long-term memory.
    """
    normalized_question = " ".join(
        question.strip().split()
    )

    if not normalized_question:
        raise ValueError(
            "Question cannot be empty."
        )

    if (
        not conversation_context.strip()
        and not user_memory_context.strip()
    ):
        return normalized_question

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing."
        )

    resolver_input = f"""
Long-term user memory:

{user_memory_context or "No saved long-term preferences."}

Recent conversation in the current session:

{conversation_context or "No recent session history."}

Latest user message:

{normalized_question}

Rewrite only the latest message as a complete standalone
GrowthGuard analytics question.
""".strip()

    resolver_agent = (
        build_context_resolver_agent()
    )

    result = await Runner.run(
        resolver_agent,
        resolver_input,
        max_turns=1,
    )

    resolved_output = result.final_output

    if not isinstance(
        resolved_output,
        ResolvedQuestion,
    ):
        return normalized_question

    standalone_question = (
        resolved_output
        .standalone_question
        .strip()
    )

    return (
        standalone_question
        or normalized_question
    )