from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MEMORY_DIRECTORY = PROJECT_ROOT / "data" / "memory"

USER_MEMORY_DATABASE_PATH = (
    MEMORY_DIRECTORY / "user_memories.db"
)

USER_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9_-]{8,128}$"
)

ALLOWED_MEMORY_KEYS = {
    "preferred_language",
    "response_style",
    "team_role",
    "primary_focus",
    "preferred_channels",
    "preferred_products",
    "recurring_goal",
}

ALLOWED_MEMORY_CATEGORIES = {
    "preference",
    "role",
    "focus",
    "goal",
}

MEMORY_LABELS = {
    "preferred_language": "Preferred language",
    "response_style": "Preferred response style",
    "team_role": "Team role",
    "primary_focus": "Primary business focus",
    "preferred_channels": "Frequently monitored channels",
    "preferred_products": "Frequently monitored products",
    "recurring_goal": "Recurring business goal",
}


def normalize_user_id(
    user_id: str,
) -> str:
    """
    Validate and normalize a persistent user identifier.
    """
    if not isinstance(user_id, str):
        raise ValueError(
            "user_id must be a string."
        )

    normalized_user_id = user_id.strip()

    if not USER_ID_PATTERN.fullmatch(
        normalized_user_id
    ):
        raise ValueError(
            "user_id must contain 8 to 128 characters "
            "using only letters, numbers, underscores, or hyphens."
        )

    return normalized_user_id


def utc_now() -> str:
    """
    Return the current UTC time as an ISO 8601 string.
    """
    return datetime.now(
        timezone.utc
    ).isoformat()


def open_database() -> sqlite3.Connection:
    """
    Open and initialize the long-term memory database.
    """
    MEMORY_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        USER_MEMORY_DATABASE_PATH,
        timeout=30,
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA journal_mode=WAL"
    )

    connection.execute(
        "PRAGMA foreign_keys=ON"
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS user_memories (
            user_id TEXT NOT NULL,
            memory_key TEXT NOT NULL,
            category TEXT NOT NULL,
            value TEXT NOT NULL,
            confidence REAL NOT NULL,
            source_session_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, memory_key)
        )
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_user_memories_user_updated
        ON user_memories (
            user_id,
            updated_at DESC
        )
        """
    )

    connection.commit()

    return connection


def normalize_memory_value(
    value: Any,
) -> str:
    """
    Normalize a stored memory value.
    """
    normalized_value = " ".join(
        str(value or "").strip().split()
    )

    return normalized_value[:500].strip()


def apply_memory_changes(
    user_id: str,
    changes: list[dict[str, Any]],
    source_session_id: str | None = None,
) -> int:
    """
    Apply structured long-term memory updates.
    """
    normalized_user_id = normalize_user_id(
        user_id
    )

    if not changes:
        return 0

    timestamp = utc_now()
    applied_count = 0

    connection = open_database()

    try:
        for change in changes:
            if not isinstance(change, dict):
                continue

            operation = str(
                change.get("operation", "")
            ).strip().lower()

            memory_key = str(
                change.get("key", "")
            ).strip().lower()

            category = str(
                change.get("category", "")
            ).strip().lower()

            value = normalize_memory_value(
                change.get("value", "")
            )

            try:
                confidence = float(
                    change.get(
                        "confidence",
                        0.0,
                    )
                )
            except (TypeError, ValueError):
                continue

            if memory_key not in ALLOWED_MEMORY_KEYS:
                continue

            if operation == "delete":
                cursor = connection.execute(
                    """
                    DELETE FROM user_memories
                    WHERE user_id = ?
                    AND memory_key = ?
                    """,
                    (
                        normalized_user_id,
                        memory_key,
                    ),
                )

                if cursor.rowcount > 0:
                    applied_count += 1

                continue

            if operation != "upsert":
                continue

            if category not in ALLOWED_MEMORY_CATEGORIES:
                continue

            if confidence < 0.70:
                continue

            if not value:
                continue

            connection.execute(
                """
                INSERT INTO user_memories (
                    user_id,
                    memory_key,
                    category,
                    value,
                    confidence,
                    source_session_id,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (
                    user_id,
                    memory_key
                )
                DO UPDATE SET
                    category = excluded.category,
                    value = excluded.value,
                    confidence = excluded.confidence,
                    source_session_id = excluded.source_session_id,
                    updated_at = excluded.updated_at
                """,
                (
                    normalized_user_id,
                    memory_key,
                    category,
                    value,
                    confidence,
                    source_session_id,
                    timestamp,
                    timestamp,
                ),
            )

            applied_count += 1

        connection.commit()

    finally:
        connection.close()

    return applied_count


def list_user_memories(
    user_id: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Return saved long-term memories for one user.
    """
    normalized_user_id = normalize_user_id(
        user_id
    )

    try:
        normalized_limit = int(limit)
    except (TypeError, ValueError):
        normalized_limit = 20

    normalized_limit = max(
        1,
        min(
            normalized_limit,
            50,
        ),
    )

    connection = open_database()

    try:
        rows = connection.execute(
            """
            SELECT
                memory_key,
                category,
                value,
                confidence,
                source_session_id,
                created_at,
                updated_at
            FROM user_memories
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (
                normalized_user_id,
                normalized_limit,
            ),
        ).fetchall()

    finally:
        connection.close()

    return [
        dict(row)
        for row in rows
    ]


def get_user_memory_context(
    user_id: str,
    limit: int = 20,
) -> str:
    """
    Format long-term memories for Agent context.
    """
    memories = list_user_memories(
        user_id=user_id,
        limit=limit,
    )

    if not memories:
        return ""

    lines: list[str] = []

    for memory in memories:
        memory_key = str(
            memory["memory_key"]
        )

        label = MEMORY_LABELS.get(
            memory_key,
            memory_key,
        )

        lines.append(
            f"- {label}: {memory['value']}"
        )

    return "\n".join(lines)


def clear_user_memories(
    user_id: str,
) -> int:
    """
    Delete every long-term memory belonging to one user.
    """
    normalized_user_id = normalize_user_id(
        user_id
    )

    connection = open_database()

    try:
        cursor = connection.execute(
            """
            DELETE FROM user_memories
            WHERE user_id = ?
            """,
            (
                normalized_user_id,
            ),
        )

        connection.commit()

        return max(
            cursor.rowcount,
            0,
        )

    finally:
        connection.close()