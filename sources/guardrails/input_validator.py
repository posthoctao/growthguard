from __future__ import annotations

import re

from sources.core.exceptions import (
    InputValidationError,
    UnsafeRequestError,
)


MAX_QUESTION_LENGTH = 3000


BLOCKED_REQUEST_PATTERNS = [
    re.compile(
        r"ignore\s+(all\s+)?previous\s+instructions",
        re.IGNORECASE,
    ),
    re.compile(
        r"(show|reveal|print|return)\s+"
        r"(the\s+)?(system|developer)\s+prompt",
        re.IGNORECASE,
    ),
    re.compile(
        r"(show|reveal|print|return)\s+"
        r"(the\s+)?api\s*key",
        re.IGNORECASE,
    ),
    re.compile(
        r"(show|reveal|print|return)\s+"
        r"(all\s+)?environment\s+variables",
        re.IGNORECASE,
    ),
    re.compile(
        r"(read|open|display)\s+"
        r"(/Users/|\.env|~/.zshrc)",
        re.IGNORECASE,
    ),
    re.compile(
        r"忽略.{0,12}(之前|以上|前面).{0,12}(指令|要求)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(显示|泄露|告诉我|输出).{0,12}"
        r"(系统提示词|开发者提示词|API.?Key|环境变量)",
        re.IGNORECASE,
    ),
]


def normalize_question(
    question: str,
) -> str:
    """
    Normalize and validate a user question.
    """
    if not isinstance(question, str):
        raise InputValidationError(
            "Question must be a string."
        )

    if "\x00" in question:
        raise InputValidationError(
            "Question contains an invalid null character."
        )

    normalized_question = " ".join(
        question.strip().split()
    )

    if not normalized_question:
        raise InputValidationError(
            "Question cannot be empty."
        )

    if len(normalized_question) > MAX_QUESTION_LENGTH:
        raise InputValidationError(
            (
                "Question exceeds the maximum length of "
                f"{MAX_QUESTION_LENGTH} characters."
            )
        )

    for pattern in BLOCKED_REQUEST_PATTERNS:
        if pattern.search(normalized_question):
            raise UnsafeRequestError(
                "Potential prompt injection or secret extraction request."
            )

    return normalized_question