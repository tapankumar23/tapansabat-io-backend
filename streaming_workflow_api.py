import logging
import os
import re
from functools import lru_cache
from typing import TypedDict

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from chat_service import _normalize_stream_part, _stringify_message_content as _stringify_content
from model_factory import DEFAULT_PROVIDER, get_chat_model

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    user_query: str
    intent: str | None
    parsed: dict | None
    result: str | None


class WorkflowRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)


class WorkflowResponse(BaseModel):
    result: str


@lru_cache(maxsize=1)
def get_llm():
    provider = os.getenv("STREAMING_WORKFLOW_PROVIDER", DEFAULT_PROVIDER)
    model_name = os.getenv("STREAMING_WORKFLOW_MODEL") or None
    return get_chat_model(provider=provider, model_name=model_name, temperature=0)


def classify(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_template(
        """
Classify the user query into one of:
- analytics
- general

Analytics: questions about shipments, hubs, counts, metrics, or anything answerable by SQL against the schema.
General: casual conversation, greetings, or anything not covered by analytics.

Query: {query}

Return only one word.
"""
    )
    response = get_llm().invoke(prompt.invoke({"query": state["user_query"]}))
    intent = response.content.strip().lower()
    if intent not in {"analytics", "general"}:
        intent = "general"
    return {**state, "intent": intent}


def metric_resolver(state: GraphState) -> GraphState:
    return state


def sql_generator(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_template(
        """
Generate a safe SQL query for PostgreSQL.

Tables:
- shipments(shipment_id, origin_hub_id, destination_hub_id, status, created_at)
- hubs(hub_id, hub_code, city)

Query: {query}

Return only SQL.
"""
    )
    response = get_llm().invoke(prompt.invoke({"query": state["user_query"]}))
    return {**state, "parsed": {"sql": response.content.strip()}}


def query_tool(_state: GraphState) -> GraphState:
    raise RuntimeError("query_tool must be called with astream_node, not ainvoke")


async def _get_db_url() -> str:
    for env_var in ("LANGGRAPH_POSTGRES_URL", "DATABASE_URL", "SUPABASE_DB_URL"):
        value = (os.getenv(env_var) or "").strip()
        if value:
            return value
    raise RuntimeError("No database URL found in environment variables.")


_READONLY_KEYWORDS = frozenset({"select", "with"})
_SQLITE_TRAGIC_PATTERNS = re.compile(
    r"\b(drop\s+table|drop\s+database|truncate|delete\s+from\s+\w+\s*;?\s*--|alter\s+table|insert\s+into|update\s+\w+\s+set|create\s+index|grant\s+on|revoke\s+on)\b",
    re.IGNORECASE,
)


def _is_safe_sql(sql: str) -> bool:
    cleaned = _strip_sql_fences(sql)
    if not cleaned:
        return False
    cleaned = cleaned.rstrip(";")
    if not cleaned:
        return False
    first_word = cleaned.split()[0].lower()
    if first_word not in _READONLY_KEYWORDS:
        return False
    if _SQLITE_TRAGIC_PATTERNS.search(cleaned):
        return False
    return True


def _strip_sql_fences(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(line for line in lines if not line.startswith("```")).strip()
    return cleaned


async def _execute_sql(sql: str) -> list[dict]:
    conn = await AsyncConnection.connect(
        await _get_db_url(),
        autocommit=True,
        prepare_threshold=None,
        row_factory=dict_row,
    )
    try:
        async with conn.cursor() as cur:
            await cur.execute(sql)
            rows = await cur.fetchall()
            return [dict(row) for row in rows]
    finally:
        await conn.close()


async def query_tool_async(state: GraphState) -> GraphState:
    sql = state["parsed"]["sql"]
    logger.info("Executing SQL: %s", sql)
    if not _is_safe_sql(sql):
        logger.warning("Blocked unsafe SQL: %s", sql)
        return {**state, "result": "SQL query blocked: only SELECT queries are allowed."}
    try:
        rows = await _execute_sql(sql)
        return {**state, "result": str(rows)}
    except Exception as exc:
        logger.exception("SQL execution failed: %s", sql)
        return {**state, "result": f"SQL execution error: {exc}"}


def formatter(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_template(
        """
Format this result into a user-friendly answer:

Query: {query}
Result: {result}
"""
    )
    response = get_llm().invoke(
        prompt.invoke(
            {
                "query": state["user_query"],
                "result": state["result"],
            }
        )
    )
    return {**state, "result": _stringify_content(response.content)}


def chat_node(state: GraphState) -> GraphState:
    response = get_llm().invoke(state["user_query"])
    return {**state, "result": _stringify_content(response.content)}


def _route(state: GraphState) -> str:
    if state["intent"] == "analytics":
        return "analytics"
    return "general"


def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("classifier", classify)
    builder.add_node("metric_resolver", metric_resolver)
    builder.add_node("sql_generator", sql_generator)
    builder.add_node("query_tool", query_tool_async)
    builder.add_node("formatter", formatter)
    builder.add_node("chat_node", chat_node)

    builder.set_entry_point("classifier")
    builder.add_conditional_edges(
        "classifier",
        _route,
        {
            "analytics": "metric_resolver",
            "general": "chat_node",
        },
    )

    builder.add_edge("metric_resolver", "sql_generator")
    builder.add_edge("sql_generator", "query_tool")
    builder.add_edge("query_tool", "formatter")
    builder.add_edge("formatter", END)

    builder.add_edge("chat_node", END)
    return builder.compile()


graph = build_graph()
router = APIRouter()


class SchemaRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)


@router.post("/v1/workflow", response_model=WorkflowResponse)
async def chat_workflow(payload: WorkflowRequest) -> WorkflowResponse:
    result = await graph.ainvoke(
        {
            "user_query": payload.query,
            "intent": None,
            "parsed": None,
            "result": None,
        }
    )
    return WorkflowResponse(result=result["result"])


@router.post("/v1/workflow/stream", response_model=None)
async def stream_workflow(request: Request, payload: WorkflowRequest) -> StreamingResponse:
    import json

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    async def event_stream():
        try:
            final_state: dict | None = None
            yield sse("start", {"query": payload.query})

            async for part in graph.astream(
                {
                    "user_query": payload.query,
                    "intent": None,
                    "parsed": None,
                    "result": None,
                },
                stream_mode=["updates", "values"],
                version="v2",
            ):
                if await request.is_disconnected():
                    return

                normalized = _normalize_stream_part(part)
                if normalized is None:
                    continue

                if normalized["type"] == "updates":
                    update_data = normalized["data"]
                    if isinstance(update_data, dict):
                        for node_name, node_update in update_data.items():
                            yield sse("node", {"node": node_name, "data": node_update})
                    continue

                if normalized["type"] == "values":
                    state = normalized["data"]
                    if isinstance(state, dict):
                        final_state = state

            if final_state is None:
                raise ValueError("Workflow completed without a final state.")

            yield sse(
                "complete",
                {
                    "result": final_state.get("result"),
                },
            )
        except ValueError as exc:
            yield sse("error", {"detail": str(exc)})
        except Exception:
            logger.exception("Unhandled streaming workflow error")
            yield sse("error", {"detail": "workflow request failed"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _run_schema_workflow(query: str) -> GraphState:
    sql = None
    prompt = ChatPromptTemplate.from_template(
        """
Generate a safe SQL query for PostgreSQL.

Tables:
- shipments(shipment_id, origin_hub_id, destination_hub_id, status, created_at)
- hubs(hub_id, hub_code, city)

Query: {query}

Return only SQL.
"""
    )
    response = get_llm().invoke(prompt.invoke({"query": query}))
    sql = response.content.strip()
    logger.info("Generated SQL: %s", sql)

    if not _is_safe_sql(sql):
        logger.warning("Blocked unsafe SQL: %s", sql)
        return {"user_query": query, "intent": "analytics", "parsed": None, "result": "SQL query blocked: only SELECT queries are allowed."}

    try:
        rows = await _execute_sql(_strip_sql_fences(sql))
        result = str(rows)
    except Exception as exc:
        logger.exception("SQL execution failed: %s", sql)
        result = f"SQL execution error: {exc}"

    prompt_fmt = ChatPromptTemplate.from_template(
        """
Format this result into a user-friendly answer:

Query: {query}
Result: {result}
"""
    )
    formatted = get_llm().invoke(prompt_fmt.invoke({"query": query, "result": result}))
    return {
        "user_query": query,
        "intent": "analytics",
        "parsed": {"sql": sql},
        "result": _stringify_content(formatted.content),
    }


@router.post("/v1/workflow/schema/stream", response_model=None)
async def stream_schema_workflow(_request: Request, payload: SchemaRequest) -> StreamingResponse:
    import json

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    async def event_stream():
        try:
            yield sse("start", {"query": payload.query})
            yield sse("node", {"node": "sql_generator", "data": "Generating SQL..."})

            result = await _run_schema_workflow(payload.query)

            yield sse("node", {"node": "query_tool", "data": {"result": result.get("result", "")}})
            yield sse("complete", {"result": result.get("result")})
        except ValueError as exc:
            yield sse("error", {"detail": str(exc)})
        except Exception:
            logger.exception("Unhandled schema workflow error")
            yield sse("error", {"detail": "workflow request failed"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


__all__ = ["router"]
