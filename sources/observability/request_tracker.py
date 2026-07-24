from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator
from uuid import uuid4

from sources.observability.logger import (
    hash_identifier,
    log_event,
)


def elapsed_milliseconds(
    started_at: float,
) -> float:
    """
    Convert elapsed monotonic time into milliseconds.
    """
    return round(
        (
            perf_counter()
            - started_at
        )
        * 1000,
        2,
    )


@dataclass
class RequestTracker:
    """
    Track one Agent request without storing its content.
    """

    request_id: str

    session_hash: str | None

    question_length: int

    started_at: float = field(
        default_factory=perf_counter
    )

    stage_durations_ms: dict[str, float] = field(
        default_factory=dict
    )

    context_resolved: bool = False

    @classmethod
    def create(
        cls,
        *,
        request_id: str | None,
        session_id: str | None,
        question_length: int,
    ) -> "RequestTracker":
        """
        Create and log a new request tracker.
        """
        tracker = cls(
            request_id=(
                request_id
                or uuid4().hex
            ),
            session_hash=hash_identifier(
                session_id
            ),
            question_length=question_length,
        )

        log_event(
            "agent_request_started",
            request_id=tracker.request_id,
            session_hash=tracker.session_hash,
            question_length=question_length,
        )

        return tracker

    @contextmanager
    def stage(
        self,
        stage_name: str,
    ) -> Iterator[None]:
        """
        Measure one workflow stage.
        """
        stage_started_at = perf_counter()

        try:
            yield

        finally:
            duration = elapsed_milliseconds(
                stage_started_at
            )

            existing_duration = (
                self.stage_durations_ms.get(
                    stage_name,
                    0.0,
                )
            )

            self.stage_durations_ms[
                stage_name
            ] = round(
                existing_duration
                + duration,
                2,
            )

    def complete(
        self,
        *,
        status: str,
        in_scope: bool,
        selected_tools: list[str],
        successful_tools: list[str],
        failed_tools: list[str],
    ) -> None:
        """
        Log successful completion of the workflow.
        """
        log_event(
            "agent_request_completed",
            request_id=self.request_id,
            session_hash=self.session_hash,
            status=status,
            in_scope=in_scope,
            context_resolved=(
                self.context_resolved
            ),
            selected_tools=selected_tools,
            successful_tools=successful_tools,
            failed_tools=failed_tools,
            stage_durations_ms=dict(
                self.stage_durations_ms
            ),
            total_duration_ms=(
                elapsed_milliseconds(
                    self.started_at
                )
            ),
        )

    def fail(
        self,
        *,
        error_code: str,
        error: Exception,
        selected_tools: list[str],
        successful_tools: list[str],
        failed_tools: list[str],
    ) -> None:
        """
        Log a failed workflow without recording sensitive content.
        """
        log_event(
            "agent_request_failed",
            level=logging.ERROR,
            request_id=self.request_id,
            session_hash=self.session_hash,
            error_code=error_code,
            error_type=type(error).__name__,
            selected_tools=selected_tools,
            successful_tools=successful_tools,
            failed_tools=failed_tools,
            stage_durations_ms=dict(
                self.stage_durations_ms
            ),
            total_duration_ms=(
                elapsed_milliseconds(
                    self.started_at
                )
            ),
        )