from __future__ import annotations

import os
import re
from typing import Literal

from agents import Agent, Runner
from pydantic import BaseModel, Field


MODEL_NAME = os.getenv(
    "OPENAI_MODEL",
    "gpt-5-nano",
)


MemoryOperation = Literal[
    "upsert",
    "delete",
]

MemoryCategory = Literal[
    "preference",
    "role",
    "focus",
    "goal",
]

MemoryKey = Literal[
    "preferred_language",
    "response_style",
    "team_role",
    "primary_focus",
    "preferred_channels",
    "preferred_products",
    "recurring_goal",
]


MEMORY_TRIGGER_PATTERNS = [
    re.compile(
        r"(记住|以后|今后|从现在开始|默认)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(我偏好|我喜欢|我主要关注|我通常关注)",
        re.IGNORECASE,
    ),
    re.compile(
        r"我是.{0,20}(团队|部门|岗位|负责人)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(忘记|不要再记|删除|清除).{0,20}"
        r"(偏好|记忆|关注点|信息)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b("
        r"remember|from now on|going forward|"
        r"i prefer|my role is|i mainly focus on|"
        r"forget|remove this preference"
        r")\b",
        re.IGNORECASE,
    ),
]


class MemoryChange(BaseModel):
    """
    One validated long-term memory change.
    """

    operation: MemoryOperation

    key: MemoryKey

    category: MemoryCategory

    value: str = Field(
        default="",
        max_length=500,
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


class MemoryExtractionResult(BaseModel):
    """
    Structured memory extraction result.
    """

    changes: list[MemoryChange] = Field(
        default_factory=list
    )


def should_extract_long_term_memory(
    user_message: str,
) -> bool:
    """
    Avoid an additional model call for ordinary KPI questions.
    """
    normalized_message = " ".join(
        user_message.strip().split()
    )

    return any(
        pattern.search(normalized_message)
        for pattern in MEMORY_TRIGGER_PATTERNS
    )


def build_memory_extractor_agent() -> Agent:
    """
    Create a restricted long-term memory extractor.
    """
    return Agent(
        name="GrowthGuard Long-Term Memory Extractor",
        model=MODEL_NAME,
        output_type=MemoryExtractionResult,
        instructions="""
Extract only durable, non-sensitive user preferences that will
remain useful across future GrowthGuard conversations.

Allowed keys:
- preferred_language
- response_style
- team_role
- primary_focus
- preferred_channels
- preferred_products
- recurring_goal

Allowed categories:
- preference
- role
- focus
- goal

Rules:
1. Only store information the user states explicitly.
2. Do not infer a long-term preference from one ordinary analytics
   question.
3. Do not store KPI values, revenue, subscriber counts, refund values,
   dates, monthly results, model answers, or other changing business
   facts.
4. Do not store passwords, API keys, addresses, medical information,
   political views, religion, ethnicity, or other sensitive attributes.
5. Use operation="upsert" when the user establishes or changes a
   durable preference.
6. Use operation="delete" when the user explicitly asks to forget a
   stored preference.
7. Use one canonical key only once.
8. Keep values concise and in the user's language.
9. Return an empty changes list when nothing should be remembered.
10. Confidence must be at least 0.70 only when the preference is
    explicit and durable.
""".strip(),
    )


async def extract_long_term_memory_changes(
    user_message: str,
    existing_memory_context: str = "",
) -> list[dict[str, object]]:
    """
    Extract validated long-term memory changes from one user message.
    """
    normalized_message = " ".join(
        user_message.strip().split()
    )

    if not normalized_message:
        return []

    if not should_extract_long_term_memory(
        normalized_message
    ):
        return []

    if not os.getenv("OPENAI_API_KEY"):
        return []

    agent_input = f"""
Existing long-term user memory:

{existing_memory_context or "No saved memories."}

Latest user message:

{normalized_message}

Return only durable memory changes explicitly supported by the
latest user message.
""".strip()

    extractor_agent = (
        build_memory_extractor_agent()
    )

    result = await Runner.run(
        extractor_agent,
        agent_input,
        max_turns=1,
    )

    extraction = result.final_output

    if not isinstance(
        extraction,
        MemoryExtractionResult,
    ):
        return []

    return [
        change.model_dump()
        for change in extraction.changes
    ]