from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Protocol
from uuid import uuid4

from langchain_core.messages import HumanMessage

from app.repositories.sessions import get_session, upsert_session
from app.graphs.chat import get_chat_app_async
from app.graphs.workflow import get_workflow_graph
from app.models.factory import get_default_model, get_default_provider
from app.utils.stream_utils import normalize_stream_part, stringify_message_content


class ChatMode(str, Enum):
    SESSION = "session"
    STATELESS = "stateless"
    ANALYTICS = "analytics"


class ChatAppFactory(Protocol):
    async def __call__(self, provider: str | None, model: str | None) -> Any: ...


class WorkflowGraphFactory(Protocol):
    def __call__(self, provider: str, model: str) -> Any: ...


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


@dataclass(frozen=True)
class WorkflowResult:
    result: str
    provider: str
    model: str


@dataclass(frozen=True)
class WorkflowStreamEvent:
    event: Literal["start", "complete", "error"]
    result: str = ""
    provider: str = ""
    model: str = ""
    detail: str = ""


async def _resolve_session_config(
    session_id: str,
    provider: str | None,
    model: str | None,
) -> tuple[str, str]:
    stored = await get_session(session_id)

    default_provider = get_default_provider().value
    default_model = get_default_model()
    if provider is not None:
        resolved_provider = provider
        resolved_model = model or (stored.model if stored else default_model)
    elif model is not None:
        resolved_provider = stored.provider if stored else default_provider
        resolved_model = model
    elif stored is not None:
        resolved_provider = stored.provider
        resolved_model = stored.model
    else:
        resolved_provider = default_provider
        resolved_model = default_model

    await upsert_session(session_id, resolved_provider, resolved_model)
    return resolved_provider, resolved_model


class ChatService:
    """Stateful session-based multi-turn chat backed by Postgres checkpointing."""

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
        cleaned, active_session_id = _normalize_input(message, session_id)
        resolved_provider, resolved_model = await _resolve_session_config(
            active_session_id, provider, model
        )
        app = await self._app_factory(provider=resolved_provider, model=resolved_model)
        config: dict = {"configurable": {"thread_id": active_session_id}}
        result = await app.ainvoke(
            {"messages": [HumanMessage(content=cleaned)]},
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
        cleaned, active_session_id = _normalize_input(message, session_id)
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
            {"messages": [HumanMessage(content=cleaned)]},
            config=config,
            stream_mode=["messages", "values"],
            version="v2",
        ):
            normalized = normalize_stream_part(part)
            if normalized is None:
                continue
            match normalized.get("type"):
                case "messages":
                    chunk, _ = normalized.get("data")
                    token = stringify_message_content(chunk.content)
                    if not token:
                        continue
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


class WorkflowService:
    """Stateless intent-classified workflow — routes to SQL analytics or general chat."""

    def __init__(self, graph_factory: WorkflowGraphFactory = get_workflow_graph) -> None:
        self._graph_factory = graph_factory

    async def run(
        self,
        *,
        query: str,
        workspace: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> WorkflowResult:
        cleaned, _ = _normalize_input(query, None)
        resolved_provider = provider or get_default_provider().value
        resolved_model = model or get_default_model(resolved_provider)
        graph = self._graph_factory(resolved_provider, resolved_model)
        result = await graph.ainvoke({"user_query": cleaned, "workspace": workspace})
        return WorkflowResult(
            result=result["result"],
            provider=resolved_provider,
            model=resolved_model,
        )

    async def stream(
        self,
        *,
        query: str,
        workspace: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> AsyncIterator[WorkflowStreamEvent]:
        cleaned, _ = _normalize_input(query, None)
        resolved_provider = provider or get_default_provider().value
        resolved_model = model or get_default_model(resolved_provider)
        graph = self._graph_factory(resolved_provider, resolved_model)

        yield WorkflowStreamEvent(
            event="start",
            provider=resolved_provider,
            model=resolved_model,
        )

        final_result = ""
        async for state in graph.astream(
            {"user_query": cleaned, "workspace": workspace},
            stream_mode="values",
        ):
            if isinstance(state, dict) and state.get("result"):
                final_result = state["result"]

        yield WorkflowStreamEvent(
            event="complete",
            result=final_result,
            provider=resolved_provider,
            model=resolved_model,
        )


def _normalize_input(message: str, session_id: str | None) -> tuple[str, str]:
    cleaned = message.strip()
    if not cleaned:
        raise ValueError("Message cannot be empty.")
    active_session_id = session_id.strip() if session_id else uuid4().hex
    return cleaned, active_session_id
