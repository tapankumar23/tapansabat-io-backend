import asyncio
import os

from psycopg import AsyncConnection
from psycopg.rows import dict_row

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

_DB_URL_ENV_VARS = ("LANGGRAPH_POSTGRES_URL", "DATABASE_URL", "SUPABASE_DB_URL")

_checkpointer_cm = None
_checkpointer: AsyncPostgresSaver | None = None
_lock: asyncio.Lock | None = None


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

        conn = await AsyncConnection.connect(
            get_database_url(),
            autocommit=True,
            prepare_threshold=None,
            row_factory=dict_row,
        )
        checkpointer = AsyncPostgresSaver(conn)
        try:
            if _auto_setup_enabled():
                await checkpointer.setup()
        except Exception:
            await conn.close()
            raise

        _checkpointer = checkpointer
        _checkpointer_cm = None


async def close() -> None:
    global _checkpointer_cm, _checkpointer

    async with _get_lock():
        if _checkpointer is None:
            _checkpointer_cm = None
            return

        await _checkpointer.conn.close()
        _checkpointer = None
        _checkpointer_cm = None
