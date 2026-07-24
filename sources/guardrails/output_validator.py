from __future__ import annotations

import re

from sources.core.exceptions import (
    OutputValidationError,
)


MAX_ANSWER_LENGTH = 20000


SECRET_PATTERNS = [
    re.compile(
        r"\bsk-[A-Za-z0-9_-]{20,}\b"
    ),
    re.compile(
        r"OPENAI_API_KEY\s*[:=]",
        re.IGNORECASE,
    ),
    re.compile(
        r"Traceback\s+\(most recent call last\)",
        re.IGNORECASE,
    ),
    re.compile(
        r"/Users/[^\s]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"sources/(agent|tools|memory|core|guardrails)/",
        re.IGNORECASE,
    ),
]


INTERNAL_MARKERS = [
    "planner agent",
    "tool_registry.py",
    "final_response_agent.py",
    "executionresult(",
    "analysisplan(",
    '"tool_results"',
    "'tool_results'",
]


def validate_final_answer(
    answer: str,
) -> str:
    """
    Validate the final user-facing answer.
    """
    if not isinstance(answer, str):
        raise OutputValidationError(
            "Final answer must be a string."
        )

    normalized_answer = answer.strip()

    if not normalized_answer:
        raise OutputValidationError(
            "Final answer is empty."
        )

    if len(normalized_answer) > MAX_ANSWER_LENGTH:
        raise OutputValidationError(
            (
                "Final answer exceeds the maximum length of "
                f"{MAX_ANSWER_LENGTH} characters."
            )
        )

    for pattern in SECRET_PATTERNS:
        if pattern.search(normalized_answer):
            raise OutputValidationError(
                "Final answer contains sensitive internal information."
            )

    lowercase_answer = normalized_answer.lower()

    for marker in INTERNAL_MARKERS:
        if marker.lower() in lowercase_answer:
            raise OutputValidationError(
                "Final answer exposes internal workflow information."
            )

    return normalized_answer