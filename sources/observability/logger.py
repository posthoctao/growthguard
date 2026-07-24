from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_LOG_FILE_PATH = (
    PROJECT_ROOT
    / "logs"
    / "agent.jsonl"
)

LOGGER_NAME = "growthguard.agent"

_CONFIGURATION_LOCK = threading.Lock()
_CONFIGURED = False


def read_positive_integer(
    environment_variable: str,
    default: int,
) -> int:
    """
    Read a positive integer from an environment variable.
    """
    raw_value = os.getenv(
        environment_variable,
        str(default),
    )

    try:
        parsed_value = int(raw_value)
    except ValueError:
        return default

    if parsed_value <= 0:
        return default

    return parsed_value


def get_log_file_path() -> Path:
    """
    Return the configured JSONL log path.
    """
    configured_path = Path(
        os.getenv(
            "AGENT_LOG_FILE",
            str(DEFAULT_LOG_FILE_PATH),
        )
    ).expanduser()

    if not configured_path.is_absolute():
        configured_path = (
            PROJECT_ROOT
            / configured_path
        )

    return configured_path.resolve()


class JsonLineFormatter(logging.Formatter):
    """
    Format every log record as one JSON object per line.
    """

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        if isinstance(record.msg, dict):
            payload: dict[str, Any] = dict(
                record.msg
            )
        else:
            payload = {
                "message": record.getMessage(),
            }

        payload.setdefault(
            "timestamp",
            datetime.now(
                timezone.utc
            ).isoformat(),
        )

        payload.setdefault(
            "level",
            record.levelname.lower(),
        )

        payload.setdefault(
            "logger",
            record.name,
        )

        if record.exc_info:
            exception_type, exception_value, _ = (
                record.exc_info
            )

            if exception_type is not None:
                payload["exception_type"] = (
                    exception_type.__name__
                )

            if exception_value is not None:
                payload["exception_message"] = str(
                    exception_value
                )

        return json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )


def configure_observability_logging(
) -> logging.Logger:
    """
    Configure the dedicated rotating JSONL logger once.
    """
    global _CONFIGURED

    with _CONFIGURATION_LOCK:
        logger = logging.getLogger(
            LOGGER_NAME
        )

        if _CONFIGURED:
            return logger

        log_file_path = get_log_file_path()

        log_file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        log_level_name = os.getenv(
            "AGENT_LOG_LEVEL",
            "INFO",
        ).upper()

        log_level = getattr(
            logging,
            log_level_name,
            logging.INFO,
        )

        logger.setLevel(log_level)
        logger.propagate = False

        has_json_handler = any(
            getattr(
                handler,
                "_growthguard_json_handler",
                False,
            )
            for handler in logger.handlers
        )

        if not has_json_handler:
            handler = RotatingFileHandler(
                filename=log_file_path,
                maxBytes=read_positive_integer(
                    "AGENT_LOG_MAX_BYTES",
                    10_000_000,
                ),
                backupCount=read_positive_integer(
                    "AGENT_LOG_BACKUP_COUNT",
                    5,
                ),
                encoding="utf-8",
            )

            handler.setLevel(log_level)

            handler.setFormatter(
                JsonLineFormatter()
            )

            setattr(
                handler,
                "_growthguard_json_handler",
                True,
            )

            logger.addHandler(handler)

        _CONFIGURED = True

        return logger


def hash_identifier(
    value: str | None,
) -> str | None:
    """
    Return a short irreversible identifier for logs.
    """
    if not value:
        return None

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:16]


def log_event(
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """
    Write one structured observability event.
    """
    logger = configure_observability_logging()

    payload = {
        "event": event,
        **fields,
    }

    logger.log(
        level,
        payload,
    )