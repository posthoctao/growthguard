from __future__ import annotations

import pytest

from sources.memory import long_term_memory


@pytest.fixture
def temporary_memory_database(
    tmp_path,
    monkeypatch,
):
    """
    Redirect long-term memory storage to a temporary database.
    """
    memory_directory = (
        tmp_path / "data" / "memory"
    )

    database_path = (
        memory_directory
        / "user_memories.db"
    )

    monkeypatch.setattr(
        long_term_memory,
        "MEMORY_DIRECTORY",
        memory_directory,
    )

    monkeypatch.setattr(
        long_term_memory,
        "USER_MEMORY_DATABASE_PATH",
        database_path,
    )

    return database_path


def test_apply_and_list_user_memories(
    temporary_memory_database,
):
    user_id = "user_test_001"

    changes = [
        {
            "operation": "upsert",
            "key": "preferred_language",
            "category": "preference",
            "value": "中文",
            "confidence": 0.95,
        },
        {
            "operation": "upsert",
            "key": "response_style",
            "category": "preference",
            "value": "先给结论",
            "confidence": 0.90,
        },
        {
            "operation": "upsert",
            "key": "primary_focus",
            "category": "focus",
            "value": "订阅留存和退款风险",
            "confidence": 0.88,
        },
    ]

    applied_count = (
        long_term_memory.apply_memory_changes(
            user_id=user_id,
            changes=changes,
            source_session_id=(
                "session_test_001"
            ),
        )
    )

    assert applied_count == 3

    memories = (
        long_term_memory.list_user_memories(
            user_id
        )
    )

    assert len(memories) == 3

    memories_by_key = {
        memory["memory_key"]: memory
        for memory in memories
    }

    assert (
        memories_by_key[
            "preferred_language"
        ]["value"]
        == "中文"
    )

    assert (
        memories_by_key[
            "response_style"
        ]["value"]
        == "先给结论"
    )

    assert (
        memories_by_key[
            "primary_focus"
        ]["value"]
        == "订阅留存和退款风险"
    )


def test_memory_context_formatting(
    temporary_memory_database,
):
    user_id = "user_test_002"

    long_term_memory.apply_memory_changes(
        user_id=user_id,
        changes=[
            {
                "operation": "upsert",
                "key": "preferred_language",
                "category": "preference",
                "value": "中文",
                "confidence": 0.95,
            }
        ],
    )

    context = (
        long_term_memory
        .get_user_memory_context(
            user_id
        )
    )

    assert "Preferred language" in context
    assert "中文" in context


def test_existing_memory_is_updated(
    temporary_memory_database,
):
    user_id = "user_test_003"

    long_term_memory.apply_memory_changes(
        user_id=user_id,
        changes=[
            {
                "operation": "upsert",
                "key": "response_style",
                "category": "preference",
                "value": "先给结论",
                "confidence": 0.85,
            }
        ],
    )

    applied_count = (
        long_term_memory.apply_memory_changes(
            user_id=user_id,
            changes=[
                {
                    "operation": "upsert",
                    "key": "response_style",
                    "category": "preference",
                    "value": (
                        "先展示关键数据，再给结论"
                    ),
                    "confidence": 0.92,
                }
            ],
        )
    )

    assert applied_count == 1

    memories = (
        long_term_memory.list_user_memories(
            user_id
        )
    )

    assert len(memories) == 1

    assert (
        memories[0]["value"]
        == "先展示关键数据，再给结论"
    )


def test_low_confidence_memory_is_rejected(
    temporary_memory_database,
):
    user_id = "user_test_004"

    applied_count = (
        long_term_memory.apply_memory_changes(
            user_id=user_id,
            changes=[
                {
                    "operation": "upsert",
                    "key": "primary_focus",
                    "category": "focus",
                    "value": "营销渠道",
                    "confidence": 0.50,
                }
            ],
        )
    )

    assert applied_count == 0

    assert (
        long_term_memory.list_user_memories(
            user_id
        )
        == []
    )


def test_delete_one_memory(
    temporary_memory_database,
):
    user_id = "user_test_005"

    long_term_memory.apply_memory_changes(
        user_id=user_id,
        changes=[
            {
                "operation": "upsert",
                "key": "preferred_language",
                "category": "preference",
                "value": "中文",
                "confidence": 0.95,
            }
        ],
    )

    deleted_count = (
        long_term_memory.apply_memory_changes(
            user_id=user_id,
            changes=[
                {
                    "operation": "delete",
                    "key": "preferred_language",
                    "category": "preference",
                    "value": "",
                    "confidence": 1.0,
                }
            ],
        )
    )

    assert deleted_count == 1

    assert (
        long_term_memory.list_user_memories(
            user_id
        )
        == []
    )


def test_clear_all_user_memories(
    temporary_memory_database,
):
    user_id = "user_test_006"

    long_term_memory.apply_memory_changes(
        user_id=user_id,
        changes=[
            {
                "operation": "upsert",
                "key": "preferred_language",
                "category": "preference",
                "value": "中文",
                "confidence": 0.95,
            },
            {
                "operation": "upsert",
                "key": "primary_focus",
                "category": "focus",
                "value": "订阅留存",
                "confidence": 0.90,
            },
        ],
    )

    deleted_count = (
        long_term_memory.clear_user_memories(
            user_id
        )
    )

    assert deleted_count == 2

    assert (
        long_term_memory.list_user_memories(
            user_id
        )
        == []
    )


@pytest.mark.parametrize(
    "invalid_user_id",
    [
        "",
        "short",
        "user id with spaces",
        "user@invalid",
    ],
)
def test_invalid_user_id_is_rejected(
    invalid_user_id,
):
    with pytest.raises(ValueError):
        long_term_memory.normalize_user_id(
            invalid_user_id
        )