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

IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9_-]{8,128}$"
)

MEMORY_LABELS = {
    "preferred_language": "回答语言",
    "response_style": "回答风格",
    "team_role": "团队角色",
    "primary_focus": "重点关注",
    "preferred_channels": "常看渠道",
    "preferred_products": "常看产品",
    "recurring_goal": "长期目标",
}


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
        min-width: 270px;
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


def normalize_identifier(
    value: Any,
) -> str | None:
    """
    Validate an identifier read from browser query parameters.
    """
    if isinstance(value, list):
        value = value[0] if value else None

    if not isinstance(value, str):
        return None

    normalized_value = value.strip()

    if not IDENTIFIER_PATTERN.fullmatch(
        normalized_value
    ):
        return None

    return normalized_value


def create_identifier() -> str:
    """
    Create a persistent browser identifier.
    """
    return uuid4().hex


def get_or_create_identifier(
    parameter_name: str,
) -> str:
    """
    Read one identifier from the URL or create it.
    """
    existing_identifier = normalize_identifier(
        st.query_params.get(
            parameter_name
        )
    )

    if existing_identifier:
        return existing_identifier

    new_identifier = create_identifier()

    st.query_params[
        parameter_name
    ] = new_identifier

    return new_identifier


def extract_api_error(
    response: requests.Response,
) -> str:
    """
    Return a safe user-facing API error message.
    """
    try:
        payload = response.json()

    except ValueError:
        payload = {}

    if isinstance(payload, dict):
        for field_name in (
            "message",
            "detail",
        ):
            value = payload.get(
                field_name
            )

            if (
                isinstance(value, str)
                and value.strip()
            ):
                return value.strip()

    if response.status_code == 404:
        return "未找到指定的数据。"

    if response.status_code == 422:
        return "请求参数无效，请检查后重试。"

    return "分析服务未能完成本次请求。"


def send_api_request(
    method: str,
    path: str,
    *,
    timeout: int = 20,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Send one request to the GrowthGuard API.
    """
    try:
        response = requests.request(
            method=method,
            url=f"{API_BASE_URL}{path}",
            json=json_body,
            timeout=timeout,
        )

    except requests.Timeout as error:
        raise RuntimeError(
            "请求超时，请稍后重试。"
        ) from error

    except requests.RequestException as error:
        raise RuntimeError(
            "分析服务暂时不可用，请确认后端已经启动。"
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

    if not isinstance(payload, dict):
        raise RuntimeError(
            "分析服务返回了无效响应。"
        )

    return payload


def normalize_history_messages(
    messages: Any,
) -> list[dict[str, str]]:
    """
    Keep valid user-facing chat messages.
    """
    if not isinstance(messages, list):
        return []

    normalized_messages: list[
        dict[str, str]
    ] = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = str(
            message.get(
                "role",
                "",
            )
        ).strip()

        content = str(
            message.get(
                "content",
                "",
            )
        ).strip()

        if (
            role not in {
                "user",
                "assistant",
            }
            or not content
        ):
            continue

        normalized_messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    return normalized_messages


def normalize_memories(
    memories: Any,
) -> list[dict[str, str]]:
    """
    Keep valid long-term memory records.
    """
    if not isinstance(memories, list):
        return []

    normalized_memories: list[
        dict[str, str]
    ] = []

    for memory in memories:
        if not isinstance(memory, dict):
            continue

        memory_key = str(
            memory.get(
                "memory_key",
                "",
            )
        ).strip()

        value = str(
            memory.get(
                "value",
                "",
            )
        ).strip()

        if not memory_key or not value:
            continue

        normalized_memories.append(
            {
                "memory_key": memory_key,
                "value": value,
            }
        )

    return normalized_memories


def fetch_session_history(
    session_id: str,
) -> tuple[
    list[dict[str, str]],
    str | None,
]:
    """
    Load short-term conversation history.
    """
    try:
        payload = send_api_request(
            "GET",
            f"/sessions/{session_id}",
        )

    except RuntimeError as error:
        return [], str(error)

    messages = normalize_history_messages(
        payload.get(
            "messages",
            [],
        )
    )

    return messages, None


def fetch_user_memories(
    user_id: str,
) -> tuple[
    list[dict[str, str]],
    str | None,
]:
    """
    Load saved long-term preferences.
    """
    try:
        payload = send_api_request(
            "GET",
            f"/users/{user_id}/memories",
        )

    except RuntimeError as error:
        return [], str(error)

    memories = normalize_memories(
        payload.get(
            "memories",
            [],
        )
    )

    return memories, None


def ask_agent(
    question: str,
    session_id: str,
    user_id: str,
) -> dict[str, str]:
    """
    Send one question with short-term and long-term IDs.
    """
    payload = send_api_request(
        "POST",
        "/ask",
        timeout=REQUEST_TIMEOUT_SECONDS,
        json_body={
            "question": question,
            "session_id": session_id,
            "user_id": user_id,
        },
    )

    answer = payload.get(
        "answer"
    )

    status = payload.get(
        "status",
        "success",
    )

    if (
        not isinstance(answer, str)
        or not answer.strip()
    ):
        raise RuntimeError(
            "分析服务未返回有效答案。"
        )

    return {
        "status": str(status),
        "answer": answer.strip(),
    }


def delete_backend_session(
    session_id: str,
) -> None:
    """
    Clear one short-term conversation.
    """
    send_api_request(
        "DELETE",
        f"/sessions/{session_id}",
    )


def delete_user_memories(
    user_id: str,
) -> int:
    """
    Clear every long-term preference for one user.
    """
    payload = send_api_request(
        "DELETE",
        f"/users/{user_id}/memories",
    )

    try:
        return max(
            int(
                payload.get(
                    "deleted_count",
                    0,
                )
            ),
            0,
        )

    except (
        TypeError,
        ValueError,
    ):
        return 0


def initialize_chat_state(
    session_id: str,
) -> None:
    """
    Load backend history when opening a session.
    """
    loaded_session_id = (
        st.session_state.get(
            "loaded_session_id"
        )
    )

    if loaded_session_id == session_id:
        return

    messages, history_error = (
        fetch_session_history(
            session_id
        )
    )

    st.session_state.messages = (
        messages
    )

    st.session_state.history_error = (
        history_error
    )

    st.session_state.loaded_session_id = (
        session_id
    )


def start_new_conversation(
    user_id: str,
) -> None:
    """
    Start a new session while preserving the same user.
    """
    new_session_id = create_identifier()

    st.query_params[
        "user_id"
    ] = user_id

    st.query_params[
        "session_id"
    ] = new_session_id

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
    Clear the current frontend and backend session.
    """
    delete_backend_session(
        session_id
    )

    st.session_state.messages = []
    st.session_state.history_error = None

    st.session_state.loaded_session_id = (
        session_id
    )

    st.rerun()


def clear_long_term_memory(
    user_id: str,
) -> None:
    """
    Clear long-term preferences and show a notice.
    """
    deleted_count = (
        delete_user_memories(
            user_id
        )
    )

    st.session_state.memory_notice = (
        f"已清除 {deleted_count} 条长期偏好。"
    )

    st.rerun()


def render_chat_history() -> None:
    """
    Render all chat messages.
    """
    for message in (
        st.session_state.messages
    ):
        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )


def render_memories(
    memories: list[dict[str, str]],
) -> None:
    """
    Render saved long-term preferences.
    """
    for memory in memories:
        label = MEMORY_LABELS.get(
            memory["memory_key"],
            memory["memory_key"],
        )

        st.caption(label)

        st.write(
            memory["value"]
        )


USER_ID = get_or_create_identifier(
    "user_id"
)

SESSION_ID = get_or_create_identifier(
    "session_id"
)

initialize_chat_state(
    SESSION_ID
)


with st.sidebar:
    st.header("对话")

    if st.button(
        "新建对话",
        use_container_width=True,
    ):
        start_new_conversation(
            USER_ID
        )

    if st.button(
        "清空当前对话",
        use_container_width=True,
    ):
        try:
            clear_current_conversation(
                SESSION_ID
            )

        except RuntimeError as error:
            st.error(
                str(error)
            )

    st.divider()
    st.subheader("长期记忆")

    memory_notice = (
        st.session_state.pop(
            "memory_notice",
            None,
        )
    )

    if memory_notice:
        st.success(
            memory_notice
        )

    memories, memory_error = (
        fetch_user_memories(
            USER_ID
        )
    )

    if memory_error:
        st.caption(
            "暂时无法读取长期记忆。"
        )

    elif memories:
        with st.expander(
            f"已保存偏好（{len(memories)}）",
            expanded=False,
        ):
            render_memories(
                memories
            )

    else:
        st.caption(
            "当前还没有保存长期偏好。"
        )

    if st.button(
        "清除长期偏好",
        use_container_width=True,
        disabled=not bool(memories),
    ):
        try:
            clear_long_term_memory(
                USER_ID
            )

        except RuntimeError as error:
            st.error(
                str(error)
            )

    st.divider()

    st.markdown(
        """
        <div class="sidebar-note">
        支持销售、订阅、客户留存、退款、网站漏斗、
        营销和产品表现分析。
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title(
    "GrowthGuard 增长分析 Agent"
)

st.markdown(
    """
    <div class="app-subtitle">
    使用自然语言分析业务表现、潜在风险和增长优先级。
    </div>
    """,
    unsafe_allow_html=True,
)


history_error = (
    st.session_state.get(
        "history_error"
    )
)

if history_error:
    st.warning(
        history_error
    )


render_chat_history()


question = st.chat_input(
    "请输入你想分析的业务问题"
)

if question:
    normalized_question = " ".join(
        question.strip().split()
    )

    if normalized_question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": normalized_question,
            }
        )

        with st.chat_message(
            "user"
        ):
            st.markdown(
                normalized_question
            )

        with st.chat_message(
            "assistant"
        ):
            with st.spinner(
                "正在分析最新可用数据……"
            ):
                try:
                    result = ask_agent(
                        question=(
                            normalized_question
                        ),
                        session_id=(
                            SESSION_ID
                        ),
                        user_id=USER_ID,
                    )

                except RuntimeError as error:
                    st.error(
                        str(error)
                    )

                else:
                    answer = result[
                        "answer"
                    ]

                    st.markdown(
                        answer
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )