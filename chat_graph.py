import asyncio
import os
import threading
from functools import lru_cache
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from model_factory import ProviderName, get_chat_model


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


_DB_URL_ENV_VARS = ("LANGGRAPH_POSTGRES_URL", "DATABASE_URL", "SUPABASE_DB_URL")
_async_checkpointer_cm = None
_async_checkpointer: AsyncPostgresSaver | None = None
_initialization_lock: asyncio.Lock | None = None
_initialization_lock_guard = threading.Lock()


def get_checkpoint_database_url() -> str:
    for env_var in _DB_URL_ENV_VARS:
        value = (os.getenv(env_var) or "").strip()
        if value:
            return value
    raise ValueError(
        "Set LANGGRAPH_POSTGRES_URL, DATABASE_URL, or SUPABASE_DB_URL to your "
        "Supabase Postgres connection string."
    )


def _should_auto_setup_checkpointer() -> bool:
    value = os.getenv("LANGGRAPH_POSTGRES_AUTO_SETUP", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _get_initialization_lock() -> asyncio.Lock:
    global _initialization_lock

    with _initialization_lock_guard:
        if _initialization_lock is None:
            _initialization_lock = asyncio.Lock()
        return _initialization_lock


async def initialize_chat_persistence() -> None:
    global _async_checkpointer_cm, _async_checkpointer

    if _async_checkpointer is not None:
        return

    async with _get_initialization_lock():
        if _async_checkpointer is not None:
            return

        database_url = get_checkpoint_database_url()
        checkpointer_cm = AsyncPostgresSaver.from_conn_string(database_url)
        checkpointer = None

        try:
            checkpointer = await checkpointer_cm.__aenter__()

            if _should_auto_setup_checkpointer():
                await checkpointer.setup()
        except Exception:
            if checkpointer is not None:
                await checkpointer_cm.__aexit__(None, None, None)
            raise

        _async_checkpointer_cm = checkpointer_cm
        _async_checkpointer = checkpointer


async def close_chat_persistence() -> None:
    global _async_checkpointer_cm, _async_checkpointer

    async with _get_initialization_lock():
        _compiled_chat_app.cache_clear()

        if _async_checkpointer_cm is None:
            _async_checkpointer = None
            return

        await _async_checkpointer_cm.__aexit__(None, None, None)
        _async_checkpointer_cm = None
        _async_checkpointer = None


def _ensure_chat_persistence_initialized() -> None:
    if _async_checkpointer is not None:
        return

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(initialize_chat_persistence())
        return

    raise RuntimeError(
        "Chat persistence is not initialized. Await initialize_chat_persistence() "
        "before using chat_graph from an async application."
    )


def _build_chat_app(
    provider: ProviderName = "openrouter",
    model_name: str | None = None,
):
    llm = get_chat_model(provider=provider, model_name=model_name)

    async def chatbot(state: ChatState) -> ChatState:
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chatbot", chatbot)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)

    return graph.compile(checkpointer=_async_checkpointer)


@lru_cache(maxsize=16)
def _compiled_chat_app(
    provider: ProviderName = "openrouter",
    model_name: str | None = None,
):
    return _build_chat_app(provider=provider, model_name=model_name)


def get_chat_app(
    provider: ProviderName = "openrouter",
    model_name: str | None = None,
):
    _ensure_chat_persistence_initialized()
    return _compiled_chat_app(provider=provider, model_name=model_name)


async def get_chat_app_async(
    provider: ProviderName = "openrouter",
    model_name: str | None = None,
):
    await initialize_chat_persistence()
    return _compiled_chat_app(provider=provider, model_name=model_name)


class LazyChatApp:
    def invoke(self, *args, **kwargs):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(*args, **kwargs))
        raise RuntimeError("Use `await app.ainvoke(...)` when running inside an event loop.")

    async def ainvoke(self, *args, **kwargs):
        chat_app = await get_chat_app_async()
        return await chat_app.ainvoke(*args, **kwargs)


app = LazyChatApp()
