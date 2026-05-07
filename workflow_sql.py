import os
import re
from collections import defaultdict

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from chat_persistence import get_database_url

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
    """Append LIMIT if absent so the DB never materialises more rows than SQL_MAX_ROWS."""
    if not re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
        return f"{sql.rstrip(';')} LIMIT {SQL_MAX_ROWS}"
    return sql


def _get_pool() -> AsyncConnectionPool:
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
    """Lightweight connectivity check — used by /readyz."""
    async with _get_pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1")


async def execute_sql(sql: str) -> list[dict]:
    """Execute a pre-validated, fence-stripped SQL query and return rows."""
    bounded_sql = _apply_row_limit(sql)
    async with _get_pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(bounded_sql)
            rows = await cur.fetchall()
            return [dict(row) for row in rows]


async def fetch_table_schema(tables: list[str]) -> str:
    """Return a live schema description string for the given table names."""
    if not tables:
        return ""
    async with _get_pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT table_name, column_name, data_type "
                "FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = ANY(%s) "
                "ORDER BY table_name, ordinal_position",
                [tables],
            )
            rows = await cur.fetchall()

    cols_by_table: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        cols_by_table[row["table_name"]].append(f"{row['column_name']} {row['data_type']}")

    return "\n".join(
        f"- {table}({', '.join(cols_by_table[table])})"
        for table in tables
        if table in cols_by_table
    )
