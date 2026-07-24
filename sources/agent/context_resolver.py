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
    A standalone question resolved from conversation history.
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
        name="GrowthGuard Conversation Context Resolver",
        model=MODEL_NAME,
        output_type=ResolvedQuestion,
        instructions="""
You resolve follow-up messages in a GrowthGuard DTC analytics
conversation.

Your only responsibility is to rewrite the user's latest message
as a complete standalone business question.

Rules:
1. Use recent conversation history only to resolve references such as:
   - it
   - that
   - those
   - them
   - what about
   - compared with that
   - 那
   - 它
   - 这个
   - 上面
   - 相比呢
   - 为什么会这样
2. Preserve the user's analytical intent.
3. Preserve explicitly mentioned dates, months, products, channels,
   metrics, and comparison requirements.
4. Do not calculate metrics.
5. Do not answer the question.
6. Do not invent business context.
7. Do not treat previously mentioned KPI values as current data.
8. Current KPI values must still be retrieved from deterministic tools.
9. Write the standalone question in the same primary language as the
   user's latest message.
10. If the latest message is already standalone, return it with only
    minimal wording cleanup.
""".strip(),
    )


async def resolve_user_question(
    question: str,
    conversation_context: str = "",
) -> str:
    """
    Resolve a user message into a standalone analytics question.
    """
    normalized_question = " ".join(
        question.strip().split()
    )

    if not normalized_question:
        raise ValueError(
            "Question cannot be empty."
        )

    if not conversation_context.strip():
        return normalized_question

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing."
        )

    resolver_input = f"""
Recent user-facing conversation:

{conversation_context}

Latest user message:

{normalized_question}

Rewrite only the latest user message as a complete standalone
GrowthGuard analytics question.
""".strip()

    resolver_agent = build_context_resolver_agent()

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