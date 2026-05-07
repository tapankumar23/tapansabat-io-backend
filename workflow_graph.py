import logging
import os
from functools import lru_cache
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from model_factory import DEFAULT_PROVIDER, get_chat_model
from stream_utils import stringify_message_content
from workflow_sql import execute_sql, fetch_table_schema, is_safe_sql, strip_sql_fences

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    user_query: str
    workspace: str
    intent: str | None
    schema_context: str | None
    parsed: dict | None
    result: str | None


# Maps each workspace name to the DB tables it may query.
# To expose a new table, add its name to the relevant list — no schema hardcoding needed.
_WORKSPACE_TABLES: dict[str, list[str]] = {
    "logistics-schema": ["shipments", "shipments_events", "lcr", "lcr_routes", "hubs"],
    "general": [],
}


@lru_cache(maxsize=1)
def get_llm():
    provider = os.getenv("STREAMING_WORKFLOW_PROVIDER", DEFAULT_PROVIDER)
    model_name = os.getenv("STREAMING_WORKFLOW_MODEL") or None
    return get_chat_model(provider=provider, model_name=model_name, temperature=0)


def classify(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_template(
        """Classify the user query into one of:
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


async def metric_resolver(state: GraphState) -> GraphState:
    """Fetch live schema for the workspace so downstream nodes don't hardcode columns."""
    tables = _WORKSPACE_TABLES.get(state["workspace"], [])
    schema = await fetch_table_schema(tables)
    return {**state, "schema_context": schema}


def sql_generator(state: GraphState) -> GraphState:
    schema_context = state.get("schema_context") or ""
    schema_section = f"Tables:\n{schema_context}" if schema_context else "(No schema available for this workspace.)"
    prompt = ChatPromptTemplate.from_template(
        """Generate a safe SQL query for PostgreSQL.

{schema_section}

Query: {query}

Return only SQL.
"""
    )
    response = get_llm().invoke(prompt.invoke({"query": state["user_query"], "schema_section": schema_section}))
    return {**state, "parsed": {"sql": response.content.strip()}}


async def query_tool_async(state: GraphState) -> GraphState:
    raw_sql = state["parsed"]["sql"]
    logger.info("Executing SQL: %s", raw_sql)
    if not is_safe_sql(raw_sql):
        logger.warning("Blocked unsafe SQL: %s", raw_sql)
        return {**state, "result": "SQL query blocked: only SELECT queries are allowed."}
    try:
        rows = await execute_sql(strip_sql_fences(raw_sql))
        return {**state, "result": str(rows)}
    except Exception as exc:
        logger.exception("SQL execution failed: %s", raw_sql)
        return {**state, "result": f"SQL execution error: {exc}"}


def formatter(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_template(
        """Format this result into a user-friendly answer:

Query: {query}
Result: {result}
"""
    )
    response = get_llm().invoke(
        prompt.invoke({"query": state["user_query"], "result": state["result"]})
    )
    return {**state, "result": stringify_message_content(response.content)}


def chat_node(state: GraphState) -> GraphState:
    response = get_llm().invoke(state["user_query"])
    return {**state, "result": stringify_message_content(response.content)}


def _route(state: GraphState) -> str:
    return "analytics" if state["intent"] == "analytics" else "general"


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
        {"analytics": "metric_resolver", "general": "chat_node"},
    )
    builder.add_edge("metric_resolver", "sql_generator")
    builder.add_edge("sql_generator", "query_tool")
    builder.add_edge("query_tool", "formatter")
    builder.add_edge("formatter", END)
    builder.add_edge("chat_node", END)

    return builder.compile()


graph = build_graph()
