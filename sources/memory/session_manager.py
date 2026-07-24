from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agents import SQLiteSession


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MEMORY_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "memory"
)

SESSION_DATABASE_PATH = (
    MEMORY_DIRECTORY
    / "conversations.db"
)

DEFAULT_HISTORY_LIMIT = 20

SESSION_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9_-]{8,128}$"
)


def normalize_session_id(
    session_id: str,
) -> str:
    """
    Validate and normalize a conversation session ID.
    """
    if not isinstance(session_id, str):
        raise ValueError(
            "session_id must be a string."
        )

    normalized_session_id = session_id.strip()

    if not SESSION_ID_PATTERN.fullmatch(
        normalized_session_id
    ):
        raise ValueError(
            "session_id must contain 8 to 128 characters "
            "using only letters, numbers, underscores, or hyphens."
        )

    return normalized_session_id


def create_session(
    session_id: str,
) -> SQLiteSession:
    """
    Create a persistent OpenAI Agents SDK SQLite session.
    """
    normalized_session_id = normalize_session_id(
        session_id
    )

    MEMORY_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    return SQLiteSession(
        session_id=normalized_session_id,
        db_path=SESSION_DATABASE_PATH,
    )


def content_to_text(
    content: Any,
) -> str:
    """
    Convert a stored Responses API message content value into text.
    """
    if isinstance(content, str):
        return content.strip()

    if not isinstance(content, list):
        return str(content).strip()

    text_parts: list[str] = []

    for item in content:
        if isinstance(item, str):
            text_parts.append(item)
            continue

        if not isinstance(item, dict):
            continue

        text_value = (
            item.get("text")
            or item.get("content")
        )

        if isinstance(text_value, str):
            text_parts.append(text_value)

    return "\n".join(
        part
        for part in text_parts
        if part
    ).strip()


async def get_session_messages(
    session_id: str,
    limit: int = DEFAULT_HISTORY_LIMIT,
) -> list[dict[str, str]]:
    """
    Return clean user and assistant messages from one session.
    """
    session = create_session(
        session_id=session_id,
    )

    try:
        items = await session.get_items(
            limit=limit,
        )

    finally:
        session.close()

    messages: list[dict[str, str]] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        role = str(
            item.get("role", "")
        ).strip()

        if role not in {
            "user",
            "assistant",
        }:
            continue

        content = content_to_text(
            item.get("content", "")
        )

        if not content:
            continue

        messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    return messages


async def get_conversation_context(
    session_id: str,
    limit: int = DEFAULT_HISTORY_LIMIT,
) -> str:
    """
    Build recent conversation context for resolving follow-up questions.
    """
    messages = await get_session_messages(
        session_id=session_id,
        limit=limit,
    )

    if not messages:
        return ""

    formatted_messages: list[str] = []

    for message in messages:
        role_label = (
            "User"
            if message["role"] == "user"
            else "Assistant"
        )

        formatted_messages.append(
            f"{role_label}: {message['content']}"
        )

    return "\n\n".join(
        formatted_messages
    )


async def save_conversation_turn(
    session_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """
    Persist one clean user-assistant conversation turn.
    """
    normalized_user_message = (
        " ".join(user_message.strip().split())
    )

    normalized_assistant_message = (
        assistant_message.strip()
    )

    if not normalized_user_message:
        raise ValueError(
            "user_message cannot be empty."
        )

    if not normalized_assistant_message:
        raise ValueError(
            "assistant_message cannot be empty."
        )

    session = create_session(
        session_id=session_id,
    )

    try:
        await session.add_items(
            [
                {
                    "role": "user",
                    "content": normalized_user_message,
                },
                {
                    "role": "assistant",
                    "content": normalized_assistant_message,
                },
            ]
        )

    finally:
        session.close()


async def clear_session_memory(
    session_id: str,
) -> None:
    """
    Delete all conversation history for one session.
    """
    session = create_session(
        session_id=session_id,
    )

    try:
        await session.clear_session()

    finally:
        session.close()