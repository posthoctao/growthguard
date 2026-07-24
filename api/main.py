from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.exceptions import (
    RequestValidationError,
)
from fastapi.middleware.cors import (
    CORSMiddleware,
)
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from sources.agent.agent_service import (
    run_agent_with_details,
)
from sources.core.exceptions import (
    AgentApplicationError,
    InputValidationError,
    RequestTimeoutError,
)
from sources.memory.session_manager import (
    clear_session_memory,
    get_session_messages,
    normalize_session_id,
)
from sources.observability.logger import (
    log_event,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_RESULT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "latest_evaluation.json"
)


REQUEST_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9_-]{8,128}$"
)


def read_positive_integer(
    environment_variable: str,
    default: int,
) -> int:
    """
    Read a positive integer environment setting.
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


REQUEST_TIMEOUT_SECONDS = (
    read_positive_integer(
        "REQUEST_TIMEOUT_SECONDS",
        180,
    )
)

MAX_CONCURRENT_REQUESTS = (
    read_positive_integer(
        "MAX_CONCURRENT_REQUESTS",
        4,
    )
)

REQUEST_SEMAPHORE = asyncio.Semaphore(
    MAX_CONCURRENT_REQUESTS
)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=3000,
    )

    session_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=128,
    )


class AskResponse(BaseModel):
    status: str
    answer: str
    session_id: str
    request_id: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: list[dict[str, str]]


app = FastAPI(
    title="GrowthGuard Growth Analytics API",
    version="1.3.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_request_id(
    supplied_request_id: str | None,
) -> str:
    """
    Validate a supplied request ID or generate a new one.
    """
    if supplied_request_id:
        normalized_request_id = (
            supplied_request_id.strip()
        )

        if REQUEST_ID_PATTERN.fullmatch(
            normalized_request_id
        ):
            return normalized_request_id

    return uuid4().hex


def get_request_id(
    request: Request,
) -> str:
    """
    Return the request ID assigned by middleware.
    """
    return getattr(
        request.state,
        "request_id",
        uuid4().hex,
    )


@app.middleware("http")
async def request_tracking_middleware(
    request: Request,
    call_next,
):
    """
    Assign a request ID and log HTTP request duration.
    """
    request_id = create_request_id(
        request.headers.get(
            "X-Request-ID"
        )
    )

    request.state.request_id = request_id

    started_at = perf_counter()
    status_code = 500

    try:
        response = await call_next(
            request
        )

        status_code = response.status_code

        response.headers[
            "X-Request-ID"
        ] = request_id

        return response

    finally:
        duration_ms = round(
            (
                perf_counter()
                - started_at
            )
            * 1000,
            2,
        )

        log_event(
            "http_request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            duration_ms=duration_ms,
        )


@app.exception_handler(
    AgentApplicationError
)
async def agent_application_error_handler(
    request: Request,
    error: AgentApplicationError,
) -> JSONResponse:
    """
    Return safe application errors.
    """
    log_event(
        "api_application_error",
        request_id=get_request_id(
            request
        ),
        error_code=error.error_code,
        status_code=error.status_code,
        error_type=type(error).__name__,
    )

    return JSONResponse(
        status_code=error.status_code,
        content={
            "status": "error",
            "error_code": (
                error.error_code
            ),
            "message": (
                error.user_message
            ),
            "request_id": (
                get_request_id(
                    request
                )
            ),
        },
    )


@app.exception_handler(
    RequestValidationError
)
async def request_validation_error_handler(
    request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    """
    Return normalized request validation errors.
    """
    log_event(
        "api_request_validation_failed",
        request_id=get_request_id(
            request
        ),
        error_code=(
            "REQUEST_VALIDATION_FAILED"
        ),
        status_code=422,
    )

    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "error_code": (
                "REQUEST_VALIDATION_FAILED"
            ),
            "message": (
                "The request body is invalid."
            ),
            "request_id": (
                get_request_id(
                    request
                )
            ),
        },
    )


@app.exception_handler(
    HTTPException
)
async def http_exception_handler(
    request: Request,
    error: HTTPException,
) -> JSONResponse:
    """
    Normalize FastAPI HTTP exceptions.
    """
    message = (
        error.detail
        if isinstance(
            error.detail,
            str,
        )
        else (
            "The request could not be completed."
        )
    )

    log_event(
        "api_http_error",
        request_id=get_request_id(
            request
        ),
        status_code=error.status_code,
    )

    return JSONResponse(
        status_code=error.status_code,
        content={
            "status": "error",
            "error_code": "HTTP_ERROR",
            "message": message,
            "request_id": (
                get_request_id(
                    request
                )
            ),
        },
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    """
    Hide unexpected internal errors from users.
    """
    log_event(
        "api_unexpected_error",
        request_id=get_request_id(
            request
        ),
        error_code="INTERNAL_ERROR",
        status_code=500,
        error_type=type(error).__name__,
    )

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_code": (
                "INTERNAL_ERROR"
            ),
            "message": (
                "The analytics service encountered "
                "an unexpected error."
            ),
            "request_id": (
                get_request_id(
                    request
                )
            ),
        },
    )


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "service": (
            "GrowthGuard Growth Analytics API"
        ),
        "status": "running",
    }


@app.get("/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "healthy",
        "openai_api_key_configured": bool(
            os.getenv(
                "OPENAI_API_KEY"
            )
        ),
        "model": os.getenv(
            "OPENAI_MODEL",
            "gpt-5-nano",
        ),
        "session_memory": (
            "persistent_sqlite"
        ),
        "guardrails": "enabled",
        "observability": (
            "rotating_jsonl"
        ),
        "tool_timeout_seconds": (
            read_positive_integer(
                "TOOL_TIMEOUT_SECONDS",
                30,
            )
        ),
        "request_timeout_seconds": (
            REQUEST_TIMEOUT_SECONDS
        ),
        "max_concurrent_requests": (
            MAX_CONCURRENT_REQUESTS
        ),
        "latest_evaluation_available": (
            EVALUATION_RESULT_PATH.exists()
        ),
    }


@app.post(
    "/ask",
    response_model=AskResponse,
)
async def ask_question(
    request_body: AskRequest,
    request: Request,
) -> AskResponse:
    session_id = (
        request_body.session_id
        or uuid4().hex
    )

    try:
        normalized_session_id = (
            normalize_session_id(
                session_id
            )
        )

    except ValueError as error:
        raise InputValidationError(
            str(error)
        ) from error

    request_id = get_request_id(
        request
    )

    async def execute_request():
        async with REQUEST_SEMAPHORE:
            return await run_agent_with_details(
                question=(
                    request_body.question
                ),
                session_id=(
                    normalized_session_id
                ),
                request_id=request_id,
            )

    try:
        result = await asyncio.wait_for(
            execute_request(),
            timeout=(
                REQUEST_TIMEOUT_SECONDS
            ),
        )

    except asyncio.TimeoutError as error:
        raise RequestTimeoutError(
            (
                "The full Agent workflow exceeded "
                f"{REQUEST_TIMEOUT_SECONDS} seconds."
            )
        ) from error

    return AskResponse(
        status=result.status,
        answer=result.answer,
        session_id=(
            normalized_session_id
        ),
        request_id=request_id,
    )


@app.get(
    "/sessions/{session_id}",
    response_model=SessionHistoryResponse,
)
async def read_session_history(
    session_id: str,
) -> SessionHistoryResponse:
    try:
        normalized_session_id = (
            normalize_session_id(
                session_id
            )
        )

    except ValueError as error:
        raise InputValidationError(
            str(error)
        ) from error

    messages = await get_session_messages(
        session_id=normalized_session_id,
    )

    return SessionHistoryResponse(
        session_id=normalized_session_id,
        messages=messages,
    )


@app.delete(
    "/sessions/{session_id}"
)
async def delete_session_history(
    session_id: str,
) -> dict[str, str]:
    try:
        normalized_session_id = (
            normalize_session_id(
                session_id
            )
        )

    except ValueError as error:
        raise InputValidationError(
            str(error)
        ) from error

    await clear_session_memory(
        session_id=normalized_session_id,
    )

    return {
        "status": "cleared",
        "session_id": (
            normalized_session_id
        ),
    }


@app.get("/evaluation/latest")
def get_latest_evaluation(
    include_cases: bool = False,
) -> dict[str, Any]:
    if not EVALUATION_RESULT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No evaluation result file was found."
            ),
        )

    try:
        with EVALUATION_RESULT_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            evaluation_data = json.load(
                file
            )

    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "The evaluation result file is invalid."
            ),
        ) from error

    if not include_cases:
        evaluation_data.pop(
            "cases",
            None,
        )

    return evaluation_data