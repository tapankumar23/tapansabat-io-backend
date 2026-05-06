import asyncio
import os
import threading

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

_DB_URL_ENV_VARS = ("LANGGRAPH_POSTGRES_URL", "DATABASE_URL", "SUPABASE_DB_URL")

_checkpointer_cm = None
_checkpointer: AsyncPostgresSaver | None = None
_lock: asyncio.Lock | None = None
_lock_guard = threading.Lock()


def get_database_url() -> str:
    for env_var in _DB_URL_ENV_VARS:
        value = (os.getenv(env_var) or "").strip()
        if value:
            return value
    raise ValueError(
        "Set LANGGRAPH_POSTGRES_URL, DATABASE_URL, or SUPABASE_DB_URL to your "
        "Supabase Postgres connection string."
    )


def get_checkpointer() -> AsyncPostgresSaver | None:
    return _checkpointer


def _auto_setup_enabled() -> bool:
    value = os.getenv("LANGGRAPH_POSTGRES_AUTO_SETUP", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _get_lock() -> asyncio.Lock:
    global _lock
    with _lock_guard:
        if _lock is None:
            _lock = asyncio.Lock()
        return _lock


async def initialize() -> None:
    global _checkpointer_cm, _checkpointer

    if _checkpointer is not None:
        return

    async with _get_lock():
        if _checkpointer is not None:
            return

        cm = AsyncPostgresSaver.from_conn_string(get_database_url())
        checkpointer = None
        try:
            checkpointer = await cm.__aenter__()
            if _auto_setup_enabled():
                await checkpointer.setup()
        except Exception:
            if checkpointer is not None:
                await cm.__aexit__(None, None, None)
            raise

        _checkpointer_cm = cm
        _checkpointer = checkpointer


async def close() -> None:
    global _checkpointer_cm, _checkpointer

    async with _get_lock():
        if _checkpointer_cm is None:
            _checkpointer = None
            return

        await _checkpointer_cm.__aexit__(None, None, None)
        _checkpointer_cm = None
        _checkpointer = None
