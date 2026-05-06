from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from uuid import uuid4

from langchain_core.messages import HumanMessage

from chat_graph import get_chat_app_async
from model_factory import DEFAULT_PROVIDER, ProviderName


class ChatAppFactory(Protocol):
    async def __call__(
        self,
        provider: ProviderName,
        model_name: str | None,
    ) -> Any: ...


@dataclass(frozen=True)
class ChatResult:
    session_id: str
    reply: str
    provider: ProviderName
    model_name: str | None = None


@dataclass(frozen=True)
class ChatStreamEvent:
    event: Literal["start", "token", "complete"]
    session_id: str
    provider: ProviderName
    model_name: str | None = None
    delta: str = ""
    reply: str = ""


class ChatService:
    def __init__(self, app_factory: ChatAppFactory = get_chat_app_async) -> None:
        self._app_factory = app_factory

    async def achat(
        self,
        *,
        message: str,
        session_id: str | None = None,
        provider: ProviderName = DEFAULT_PROVIDER,
        model_name: str | None = None,
    ) -> ChatResult:
        cleaned_message, active_session_id = _normalize_chat_input(
            message=message,
            session_id=session_id,
        )
        app = await self._app_factory(provider=provider, model_name=model_name)
        result = await app.ainvoke(
            {"messages": [HumanMessage(content=cleaned_message)]},
            config={"configurable": {"thread_id": active_session_id}},
        )

        return ChatResult(
            session_id=active_session_id,
            reply=_stringify_message_content(result["messages"][-1].content),
            provider=provider,
            model_name=model_name,
        )

    async def astream(
        self,
        *,
        message: str,
        session_id: str | None = None,
        provider: ProviderName = DEFAULT_PROVIDER,
        model_name: str | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        cleaned_message, active_session_id = _normalize_chat_input(
            message=message,
            session_id=session_id,
        )
        app = await self._app_factory(provider=provider, model_name=model_name)
        final_reply = ""

        yield ChatStreamEvent(
            event="start",
            session_id=active_session_id,
            provider=provider,
            model_name=model_name,
        )

        async for part in app.astream(
            {"messages": [HumanMessage(content=cleaned_message)]},
            config={"configurable": {"thread_id": active_session_id}},
            stream_mode=["messages", "values"],
            version="v2",
        ):
            part = _normalize_stream_part(part)
            if part is None:
                continue

            if part.get("type") == "messages":
                message_chunk, _metadata = part["data"]
                delta = _stringify_message_content(message_chunk.content)
                if delta:
                    yield ChatStreamEvent(
                        event="token",
                        session_id=active_session_id,
                        provider=provider,
                        model_name=model_name,
                        delta=delta,
                    )
                continue

            if part.get("type") != "values":
                continue

            state = part.get("data")
            if not isinstance(state, dict):
                continue

            messages = state.get("messages")
            if isinstance(messages, list) and messages:
                final_reply = _stringify_message_content(messages[-1].content)

        yield ChatStreamEvent(
            event="complete",
            session_id=active_session_id,
            provider=provider,
            model_name=model_name,
            reply=final_reply,
        )


def _normalize_stream_part(part: object) -> dict[str, object] | None:
    if isinstance(part, dict):
        return part

    if (
        isinstance(part, tuple)
        and len(part) == 2
        and isinstance(part[0], str)
    ):
        stream_type, data = part
        return {"type": stream_type, "data": data}

    return None


def _normalize_chat_input(*, message: str, session_id: str | None) -> tuple[str, str]:
    cleaned_message = message.strip()
    if not cleaned_message:
        raise ValueError("message must not be empty")
    active_session_id = (session_id or "").strip() or uuid4().hex
    return cleaned_message, active_session_id


def _stringify_message_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        if text_parts:
            return "\n".join(text_parts)
    return str(content)
