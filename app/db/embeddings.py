import uuid
from dataclasses import dataclass

import httpx
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.models.factory import get_embedding_config, get_embedding_dimension, get_embedding_model


@dataclass
class TableEmbedding:
    id: str
    workspace: str
    table_name: str
    column_names: str
    description: str
    embedding: list[float]


def _get_pool() -> AsyncConnectionPool:
    from app.db.sql import get_pool
    return get_pool()


async def _get_embedding(text: str) -> list[float]:
    api_key, base_url, _ = get_embedding_config()
    model = get_embedding_model()
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{base_url}/embeddings",
            json={"model": model, "input": text},
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]


async def ensure_schema_table() -> None:
    dim = get_embedding_dimension()
    pool = _get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DROP TABLE IF EXISTS schema_embeddings")
            await cur.execute(
                f"""
                CREATE TABLE schema_embeddings (
                    id TEXT PRIMARY KEY,
                    workspace TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    column_names TEXT NOT NULL,
                    description TEXT NOT NULL,
                    embedding vector({dim}) NOT NULL
                )
                """
            )
            await cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_schema_embeddings_workspace "
                "ON schema_embeddings (workspace)"
            )


async def upsert_table_embedding(
    workspace: str,
    table_name: str,
    column_names: list[str],
    description: str,
) -> TableEmbedding:
    text = f"Table {table_name} in workspace {workspace}. Columns: {', '.join(column_names)}. {description}"
    embedding = await _get_embedding(text)
    record_id = str(uuid.uuid4())
    pool = _get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO schema_embeddings (id, workspace, table_name, column_names, description, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    workspace = EXCLUDED.workspace,
                    table_name = EXCLUDED.table_name,
                    column_names = EXCLUDED.column_names,
                    description = EXCLUDED.description,
                    embedding = EXCLUDED.embedding
                """,
                (record_id, workspace, table_name, ", ".join(column_names), description, embedding),
            )
    return TableEmbedding(
        id=record_id,
        workspace=workspace,
        table_name=table_name,
        column_names=", ".join(column_names),
        description=description,
        embedding=embedding,
    )


async def query_similar_tables(
    workspace: str,
    query: str,
    top_k: int = 3,
) -> list[TableEmbedding]:
    query_embedding = await _get_embedding(query)
    pool = _get_pool()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT id, workspace, table_name, column_names, description
                FROM schema_embeddings
                WHERE workspace = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (workspace, query_embedding, top_k),
            )
            rows = await cur.fetchall()
    return [
        TableEmbedding(
            id=row["id"],
            workspace=row["workspace"],
            table_name=row["table_name"],
            column_names=row["column_names"],
            description=row["description"],
            embedding=query_embedding,
        )
        for row in rows
    ]


async def clear_workspace_embeddings(workspace: str) -> None:
    pool = _get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM schema_embeddings WHERE workspace = %s",
                (workspace,),
            )
