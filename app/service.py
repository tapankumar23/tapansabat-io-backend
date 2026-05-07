from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import uuid4

from langchain_core.messages import HumanMessage

from app.api.sessions import SessionConfig, get_session, upsert_session
from app.graphs.chat import get_chat_app_async
from app.utils.stream_utils import stringify_message_content


class ChatAppFactory(Protocol):
    async def __call__(self, provider: str | None, model: str | None) -> Protocol: ...


@dataclass(frozen=True)
class ChatResult:
    session_id: str
    reply: str
    provider: str
    model: str


@dataclass(frozen=True)
class ChatStreamEvent:
    event: Literal["start", "token", "complete"]
    session_id: str
    delta: str = ""
    reply: str = ""
    provider: str = ""
    model: str = ""


def normalize_stream_part(part: object) -> dict[str, object] | None:
    if isinstance(part, dict):
        return part
    if (isinstance(part, tuple) and len(part) == 2 and isinstance(part[0], str)):
        stream_type, data = part
        return {"type": stream_type, "data": data}
    return None


async def _resolve_session_config(
    session_id: str,
    provider: str | None,
    model: str | None,
) -> tuple[str, str]:
    stored = await get_session(session_id)

    if provider is not None:
        resolved_provider = provider
        resolved_model = model or (stored.model if stored else "openrouter/free")
    elif model is not None:
        resolved_provider = stored.provider if stored else "openrouter"
        resolved_model = model
    elif stored is not None:
        resolved_provider = stored.provider
        resolved_model = stored.model
    else:
        resolved_provider = "openrouter"
        resolved_model = "openrouter/free"

    await upsert_session(session_id, resolved_provider, resolved_model)
    return resolved_provider, resolved_model


class ChatService:
    def __init__(self, app_factory: ChatAppFactory = get_chat_app_async) -> None:
        self._app_factory = app_factory

    async def achat(
        self,
        *,
        message: str,
        session_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> ChatResult:
        cleaned_message, active_session_id = _normalize_chat_input(
            message=message,
            session_id=session_id,
        )

        resolved_provider, resolved_model = await _resolve_session_config(
            active_session_id, provider, model
        )

        app = await self._app_factory(provider=resolved_provider, model=resolved_model)
        config: dict = {"configurable": {"thread_id": active_session_id}}
        result = await app.ainvoke(
            {"messages": [HumanMessage(content=cleaned_message)]},
            config=config,
        )

        return ChatResult(
            session_id=active_session_id,
            reply=stringify_message_content(result["messages"][-1].content),
            provider=resolved_provider,
            model=resolved_model,
        )

    async def astream(
        self,
        *,
        message: str,
        session_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        cleaned_message, active_session_id = _normalize_chat_input(
            message=message,
            session_id=session_id,
        )

        resolved_provider, resolved_model = await _resolve_session_config(
            active_session_id, provider, model
        )

        app = await self._app_factory(provider=resolved_provider, model=resolved_model)
        config: dict = {"configurable": {"thread_id": active_session_id}}
        final_reply = ""

        yield ChatStreamEvent(
            event="start",
            session_id=active_session_id,
            provider=resolved_provider,
            model=resolved_model,
        )

        async for part in app.astream(
            {"messages": [HumanMessage(content=cleaned_message)]},
            config=config,
            stream_mode=["messages", "values"],
            version="v2",
        ):
            normalized = normalize_stream_part(part)
            if normalized is None:
                continue
            match normalized.get("type"):
                case "messages/iter":
                    token = stringify_message_content(normalized.get("data", ""))
                    final_reply += token
                    yield ChatStreamEvent(
                        event="token",
                        session_id=active_session_id,
                        delta=token,
                        provider=resolved_provider,
                        model=resolved_model,
                    )
                case "values":
                    val = normalized.get("data", {})
                    msgs = val.get("messages", [])
                    if msgs:
                        final_reply = stringify_message_content(msgs[-1].content)

        yield ChatStreamEvent(
            event="complete",
            session_id=active_session_id,
            reply=final_reply,
            provider=resolved_provider,
            model=resolved_model,
        )


def _normalize_chat_input(*, message: str, session_id: str | None) -> tuple[str, str]:
    cleaned = message.strip()
    if not cleaned:
        raise ValueError("Message cannot be empty.")
    active_session_id = session_id.strip() if session_id else uuid4().hex
    return cleaned, active_session_id
