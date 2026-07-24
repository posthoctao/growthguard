from __future__ import annotations


class AgentApplicationError(Exception):
    """
    Base exception for safe user-facing application errors.
    """

    error_code = "APPLICATION_ERROR"
    status_code = 500
    user_message = (
        "The analytics service could not complete the request."
    )

    def __init__(
        self,
        internal_message: str | None = None,
    ) -> None:
        super().__init__(
            internal_message
            or self.user_message
        )


class InputValidationError(
    AgentApplicationError
):
    error_code = "INVALID_INPUT"
    status_code = 422
    user_message = (
        "The submitted question is invalid."
    )


class UnsafeRequestError(
    AgentApplicationError
):
    error_code = "UNSAFE_REQUEST"
    status_code = 400
    user_message = (
        "This request cannot be processed."
    )


class PlanningError(
    AgentApplicationError
):
    error_code = "PLANNING_FAILED"
    status_code = 500
    user_message = (
        "The analysis plan could not be created."
    )


class ToolExecutionError(
    AgentApplicationError
):
    error_code = "TOOL_EXECUTION_FAILED"
    status_code = 500
    user_message = (
        "The requested analysis could not be completed "
        "with the available data."
    )


class OutputValidationError(
    AgentApplicationError
):
    error_code = "OUTPUT_VALIDATION_FAILED"
    status_code = 500
    user_message = (
        "The analysis was completed, but the response "
        "could not be safely returned."
    )


class UpstreamServiceError(
    AgentApplicationError
):
    error_code = "UPSTREAM_SERVICE_ERROR"
    status_code = 503
    user_message = (
        "The AI service is temporarily unavailable."
    )


class RequestTimeoutError(
    AgentApplicationError
):
    error_code = "REQUEST_TIMEOUT"
    status_code = 504
    user_message = (
        "The analytics request took too long to complete."
    )