import os
import re

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.models.persistence import get_database_url

SQL_MAX_ROWS: int = int(os.getenv("SQL_MAX_ROWS", "500"))

_pool: AsyncConnectionPool | None = None

_READONLY_KEYWORDS = frozenset({"select", "with"})
_UNSAFE_PATTERNS = re.compile(
    r"\b(drop\s+table|drop\s+database|truncate|delete\s+from\s+\w+\s*;?\s*--|alter\s+table|insert\s+into|update\s+\w+\s+set|create\s+index|grant\s+on|revoke\s+on)\b",
    re.IGNORECASE,
)


def strip_sql_fences(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(line for line in lines if not line.startswith("```")).strip()
    return cleaned


def is_safe_sql(sql: str) -> bool:
    cleaned = strip_sql_fences(sql).rstrip(";")
    if not cleaned:
        return False
    tokens = cleaned.split()
    if not tokens or tokens[0].lower() not in _READONLY_KEYWORDS:
        return False
    return not _UNSAFE_PATTERNS.search(cleaned)


def _apply_row_limit(sql: str) -> str:
    if not re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
        return f"{sql.rstrip(';')} LIMIT {SQL_MAX_ROWS}"
    return sql


def get_pool() -> AsyncConnectionPool:
    if _pool is None:
        raise RuntimeError(
            "SQL connection pool is not initialised. "
            "Ensure open_pool() is called during application startup."
        )
    return _pool


async def open_pool() -> None:
    global _pool
    if _pool is not None:
        return
    _pool = AsyncConnectionPool(
        conninfo=get_database_url(),
        min_size=1,
        max_size=int(os.getenv("SQL_POOL_SIZE", "5")),
        kwargs={"autocommit": True, "prepare_threshold": None, "row_factory": dict_row},
        open=False,
    )
    await _pool.open()


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def ping_db() -> None:
    async with get_pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1")


async def execute_sql(sql: str) -> list[dict]:
    capped_sql = _apply_row_limit(sql)
    async with get_pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(capped_sql)
            rows = await cur.fetchall()
            return [dict(row) for row in rows]


_WORKSPACE_SCHEMA_CACHE: dict[str, str] = {}


async def fetch_table_schema(tables: list[str]) -> str:
    if not tables:
        return ""
    cache_key = ",".join(tables)
    if cache_key in _WORKSPACE_SCHEMA_CACHE:
        return _WORKSPACE_SCHEMA_CACHE[cache_key]
    schema_lines: list[str] = []
    for table in tables:
        async with get_pool().connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = %s "
                    "ORDER BY ordinal_position",
                    (table,),
                )
                cols = await cur.fetchall()
        if cols:
            col_defs = ", ".join(f"{r['column_name']} {r['data_type']}" for r in cols)
            schema_lines.append(f"{table}: {col_defs}")
    schema = "\n".join(schema_lines)
    _WORKSPACE_SCHEMA_CACHE[cache_key] = schema
    return schema
