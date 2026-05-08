import logging
from functools import lru_cache
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from app.models.factory import ProviderName, get_default_model, get_default_provider, get_llm
from app.db.sql import execute_sql, fetch_table_schema, is_safe_sql, strip_sql_fences
from app.prompts import CLASSIFY_INTENT, FORMAT_RESULT, SQL_GENERATE
from app.utils.stream_utils import stringify_message_content

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    user_query: str
    workspace: str
    intent: str | None
    schema_context: str | None
    parsed: dict | None
    result: str | None


_WORKSPACE_TABLES: dict[str, list[str]] = {
    "logistics-schema": ["shipments", "shipments_events", "lcr", "lcr_routes", "hubs"],
    "general": [],
}


@lru_cache(maxsize=None)
def _get_llm(provider: str, model: str):
    return get_llm(provider=ProviderName(provider), model=model, temperature=0)


def classify(state: GraphState, llm) -> GraphState:
    prompt = ChatPromptTemplate.from_template(CLASSIFY_INTENT)
    response = llm.invoke(prompt.invoke({"query": state["user_query"]}))
    intent = response.content.strip().lower()
    if intent not in {"analytics", "general"}:
        intent = "general"
    return {**state, "intent": intent}


async def metric_resolver(state: GraphState) -> GraphState:
    tables = _WORKSPACE_TABLES.get(state["workspace"], [])
    schema = await fetch_table_schema(tables)
    return {**state, "schema_context": schema}


def sql_generator(state: GraphState, llm) -> GraphState:
    schema_context = state.get("schema_context") or ""
    schema_section = f"Tables:\n{schema_context}" if schema_context else "(No schema available for this workspace.)"
    prompt = ChatPromptTemplate.from_template(SQL_GENERATE)
    response = llm.invoke(prompt.invoke({"query": state["user_query"], "schema_section": schema_section}))
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


def formatter(state: GraphState, llm) -> GraphState:
    prompt = ChatPromptTemplate.from_template(FORMAT_RESULT)
    response = llm.invoke(
        prompt.invoke({"query": state["user_query"], "result": state["result"]})
    )
    return {**state, "result": stringify_message_content(response.content)}


def chat_node(state: GraphState, llm) -> GraphState:
    response = llm.invoke(state["user_query"])
    return {**state, "result": stringify_message_content(response.content)}


_INTENT_TO_BRANCH: dict[str, str] = {
    "analytics": "analytics",
}


def _route(state: GraphState) -> str:
    return _INTENT_TO_BRANCH.get(state["intent"] or "", "general")


def build_graph(provider: str, model: str):
    sql_llm = _get_llm(provider, model)
    chat_llm = _get_llm(provider, model)

    def classify_with_llm(state: GraphState) -> GraphState:
        return classify(state, sql_llm)

    def sql_generator_with_llm(state: GraphState) -> GraphState:
        return sql_generator(state, sql_llm)

    def formatter_with_llm(state: GraphState) -> GraphState:
        return formatter(state, sql_llm)

    def chat_node_with_llm(state: GraphState) -> GraphState:
        return chat_node(state, chat_llm)

    builder = StateGraph(GraphState)
    builder.add_node("classifier", classify_with_llm)
    builder.add_node("metric_resolver", metric_resolver)
    builder.add_node("sql_generator", sql_generator_with_llm)
    builder.add_node("query_tool", query_tool_async)
    builder.add_node("formatter", formatter_with_llm)
    builder.add_node("chat_node", chat_node_with_llm)

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


@lru_cache(maxsize=None)
def _compiled_workflow_graph(provider: str, model: str):
    return build_graph(provider, model)


def get_workflow_graph(provider: str, model: str):
    return _compiled_workflow_graph(provider, model)


graph = build_graph(get_default_provider().value, get_default_model())


