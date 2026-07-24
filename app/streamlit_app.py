from __future__ import annotations

import os
import re
from typing import Any
from uuid import uuid4

import requests
import streamlit as st


API_BASE_URL = os.getenv(
    "GROWTHGUARD_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

REQUEST_TIMEOUT_SECONDS = 180

SESSION_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9_-]{8,128}$"
)


st.set_page_config(
    page_title="GrowthGuard 增长分析 Agent",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .block-container {
            max-width: 900px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            min-width: 260px;
        }

        .app-subtitle {
            color: #6b7280;
            font-size: 0.98rem;
            margin-top: -0.7rem;
            margin-bottom: 1.8rem;
        }

        .sidebar-note {
            color: #6b7280;
            font-size: 0.88rem;
            line-height: 1.5;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_session_id(
    session_id: Any,
) -> str | None:
    """
    Validate a session ID read from the browser URL.
    """
    if isinstance(session_id, list):
        session_id = (
            session_id[0]
            if session_id
            else None
        )

    if not isinstance(session_id, str):
        return None

    normalized_session_id = session_id.strip()

    if not SESSION_ID_PATTERN.fullmatch(
        normalized_session_id
    ):
        return None

    return normalized_session_id


def create_new_session_id() -> str:
    """
    Generate a new persistent conversation ID.
    """
    return uuid4().hex


def get_or_create_session_id() -> str:
    """
    Read the session ID from the browser URL or create a new one.
    """
    existing_session_id = normalize_session_id(
        st.query_params.get("session_id")
    )

    if existing_session_id:
        return existing_session_id

    new_session_id = create_new_session_id()

    st.query_params["session_id"] = (
        new_session_id
    )

    return new_session_id


def extract_api_error(
    response: requests.Response,
) -> str:
    """Return a user-facing API error message."""
    try:
        payload = response.json()
    except ValueError:
        payload = {}

    if isinstance(payload, dict):
        for field_name in ("detail", "message"):
            value = payload.get(field_name)

            if isinstance(value, str) and value.strip():
                return value.strip()

    if response.status_code == 404:
        return "未找到指定的对话。"

    if response.status_code == 422:
        return "当前问题无法处理，请检查输入后重试。"

    return "分析服务未能完成本次请求。"

def normalize_history_messages(
    messages: Any,
) -> list[dict[str, str]]:
    """
    Keep only valid user-facing chat messages.
    """
    if not isinstance(messages, list):
        return []

    normalized_messages: list[dict[str, str]] = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = str(
            message.get("role", "")
        ).strip()

        content = str(
            message.get("content", "")
        ).strip()

        if role not in {
            "user",
            "assistant",
        }:
            continue

        if not content:
            continue

        normalized_messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    return normalized_messages


def fetch_session_history(
    session_id: str,
) -> tuple[list[dict[str, str]], str | None]:
    """
    Load persistent conversation history from FastAPI.
    """
    try:
        response = requests.get(
            (
                f"{API_BASE_URL}"
                f"/sessions/{session_id}"
            ),
            timeout=20,
        )

    except requests.RequestException:
        return (
            [],
            (
                "无法加载历史消息。 "
                "请确认后端服务已经启动。"
            ),
        )

    if not response.ok:
        return (
            [],
            extract_api_error(response),
        )

    try:
        payload = response.json()
    except ValueError:
        return (
            [],
            "对话历史响应格式无效。",
        )

    messages = normalize_history_messages(
        payload.get("messages", [])
    )

    return messages, None


def ask_agent(
    question: str,
    session_id: str,
) -> dict[str, str]:
    """
    Send one question to the persistent-memory Agent API.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={
                "question": question,
                "session_id": session_id,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    except requests.Timeout as error:
        raise RuntimeError(
            "请求超时，请稍后重试。"
        ) from error

    except requests.RequestException as error:
        raise RuntimeError(
            (
                "分析服务暂时不可用。 "
                "请确认后端服务已经启动。"
            )
        ) from error

    if not response.ok:
        raise RuntimeError(
            extract_api_error(response)
        )

    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError(
            "分析服务返回了无效响应。"
        ) from error

    answer = payload.get("answer")
    status = payload.get("status", "success")

    if not isinstance(answer, str):
        raise RuntimeError(
            "分析服务未返回答案。"
        )

    answer = answer.strip()

    if not answer:
        raise RuntimeError(
            "分析服务返回了空答案。"
        )

    return {
        "status": str(status),
        "answer": answer,
    }


def delete_backend_session(
    session_id: str,
) -> None:
    """
    Delete all stored messages from one backend session.
    """
    try:
        response = requests.delete(
            (
                f"{API_BASE_URL}"
                f"/sessions/{session_id}"
            ),
            timeout=20,
        )

    except requests.RequestException as error:
        raise RuntimeError(
            (
                "无法清空当前对话。"
                "请确认后端服务已经启动。"
            )
        ) from error

    if not response.ok:
        raise RuntimeError(
            extract_api_error(response)
        )


def initialize_chat_state(
    session_id: str,
) -> None:
    """
    Load backend history when the browser opens a session.
    """
    loaded_session_id = st.session_state.get(
        "loaded_session_id"
    )

    if loaded_session_id == session_id:
        return

    messages, history_error = (
        fetch_session_history(
            session_id=session_id,
        )
    )

    st.session_state.messages = messages
    st.session_state.history_error = (
        history_error
    )
    st.session_state.loaded_session_id = (
        session_id
    )


def start_new_conversation() -> None:
    """
    Start a new conversation without deleting previous sessions.
    """
    new_session_id = create_new_session_id()

    st.query_params["session_id"] = (
        new_session_id
    )

    st.session_state.messages = []
    st.session_state.history_error = None
    st.session_state.loaded_session_id = (
        new_session_id
    )

    st.rerun()


def clear_current_conversation(
    session_id: str,
) -> None:
    """
    Clear the current frontend and backend conversation.
    """
    delete_backend_session(
        session_id=session_id,
    )

    st.session_state.messages = []
    st.session_state.history_error = None
    st.session_state.loaded_session_id = (
        session_id
    )

    st.rerun()


def render_chat_history() -> None:
    """
    Render all user and assistant messages.
    """
    for message in st.session_state.messages:
        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )


SESSION_ID = get_or_create_session_id()

initialize_chat_state(
    session_id=SESSION_ID,
)


with st.sidebar:
    st.header("对话")

    if st.button(
        "新建对话",
        use_container_width=True,
    ):
        start_new_conversation()

    if st.button(
        "清空当前对话",
        use_container_width=True,
    ):
        try:
            clear_current_conversation(
                session_id=SESSION_ID,
            )

        except RuntimeError as error:
            st.error(str(error))

    st.divider()

    st.markdown(
        """
        <div class="sidebar-note">
            支持销售、订阅、客户留存、退款、网站漏斗、营销和产品表现分析。
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title("GrowthGuard 增长分析 Agent")

st.markdown(
    """
    <div class="app-subtitle">
        使用自然语言分析业务表现、潜在风险和增长优先级。
    </div>
    """,
    unsafe_allow_html=True,
)


history_error = st.session_state.get(
    "history_error"
)

if history_error:
    st.warning(history_error)


render_chat_history()


question = st.chat_input(
    "请输入你想分析的业务问题"
)


if question:
    normalized_question = " ".join(
        question.strip().split()
    )

    if normalized_question:
        user_message = {
            "role": "user",
            "content": normalized_question,
        }

        st.session_state.messages.append(
            user_message
        )

        with st.chat_message("user"):
            st.markdown(
                normalized_question
            )

        with st.chat_message("assistant"):
            with st.spinner(
                "正在分析最新可用数据……"
            ):
                try:
                    result = ask_agent(
                        question=normalized_question,
                        session_id=SESSION_ID,
                    )

                except RuntimeError as error:
                    st.error(str(error))

                else:
                    answer = result["answer"]

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )