from __future__ import annotations

from pydantic import BaseModel

from app.db.sql import get_pool


class SessionConfig(BaseModel):
    session_id: str
    provider: str = "openrouter"
    model: str = "openrouter/free"


async def ensure_sessions_table() -> None:
    async with get_pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id VARCHAR(128) PRIMARY KEY,
                    provider   VARCHAR(32)  NOT NULL DEFAULT 'openrouter',
                    model      VARCHAR(64)  NOT NULL DEFAULT 'openrouter/free',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )


async def get_session(session_id: str) -> SessionConfig | None:
    async with get_pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT session_id, provider, model FROM chat_sessions WHERE session_id = %s",
                (session_id,),
            )
            row = await cur.fetchone()
    if row is None:
        return None
    return SessionConfig(session_id=row["session_id"], provider=row["provider"], model=row["model"])


async def upsert_session(session_id: str, provider: str, model: str) -> SessionConfig:
    async with get_pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO chat_sessions (session_id, provider, model, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (session_id) DO UPDATE
                    SET provider = EXCLUDED.provider,
                        model   = EXCLUDED.model,
                        updated_at = NOW()
                RETURNING session_id, provider, model
                """,
                (session_id, provider, model),
            )
            row = await cur.fetchone()
    return SessionConfig(session_id=row["session_id"], provider=row["provider"], model=row["model"])


async def delete_session(session_id: str) -> bool:
    async with get_pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM chat_sessions WHERE session_id = %s",
                (session_id,),
            )
            deleted = cur.rowcount
    return (deleted or 0) > 0
