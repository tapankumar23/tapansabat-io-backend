import json
import logging
import os
from functools import lru_cache
from typing import Optional, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    user_query: str
    intent: Optional[str]
    parsed: Optional[dict]
    result: Optional[str]


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "streaming-workflow-api"


class ErrorResponse(BaseModel):
    detail: str


class WorkflowRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)


class WorkflowResponse(BaseModel):
    intent: str | None = None
    result: str


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    model_name = os.getenv("STREAMING_WORKFLOW_MODEL", "gpt-4o-mini")
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("STREAMING_WORKFLOW_BASE_URL") or "").strip()

    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for `streaming_workflow_api.py`.")

    kwargs: dict[str, object] = {"model": model_name, "temperature": 0}
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(api_key=api_key, **kwargs)


def classify(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_template(
        """
Classify the user query into one of:
- analytics
- action
- chat

Query: {query}

Return only one word.
"""
    )
    response = get_llm().invoke(prompt.invoke({"query": state["user_query"]}))
    intent = response.content.strip().lower()
    if intent not in {"analytics", "action", "chat"}:
        intent = "chat"
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


def query_tool(state: GraphState) -> GraphState:
    sql = state["parsed"]["sql"]

    logger.info("Executing SQL: %s", sql)

    result = [{"hub": "BLR", "count": 120}]
    return {**state, "result": json.dumps(result)}


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


def intent_parser(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_template(
        """
Convert this into JSON for route update.

Fields:
- action
- source_hub
- destination_hub
- current_hub
- new_next_hub
- sequence_no

Query: {query}

Return valid JSON only.
"""
    )
    response = get_llm().invoke(prompt.invoke({"query": state["user_query"]}))
    parsed = _parse_json_object(_stringify_content(response.content))
    return {**state, "parsed": parsed}


def validate(state: GraphState) -> GraphState:
    payload = state["parsed"] or {}
    if payload.get("current_hub") == payload.get("new_next_hub"):
        raise ValueError("Invalid route update")
    return state


def update_lcr_route_api(payload: dict) -> dict:
    logger.info("Updating route: %s", payload)
    return {"status": "success"}


def tool_caller(state: GraphState) -> GraphState:
    payload = state["parsed"] or {}
    response = update_lcr_route_api(payload)
    return {**state, "result": json.dumps(response)}


def action_formatter(state: GraphState) -> GraphState:
    payload = state["parsed"] or {}
    source_hub = payload.get("source_hub", "unknown")
    destination_hub = payload.get("destination_hub", "unknown")
    return {
        **state,
        "result": f"Route updated: {source_hub} -> {destination_hub}",
    }


def chat_node(state: GraphState) -> GraphState:
    response = get_llm().invoke(state["user_query"])
    return {**state, "result": _stringify_content(response.content)}


def router(state: GraphState) -> str:
    if state["intent"] == "analytics":
        return "analytics"
    if state["intent"] == "action":
        return "action"
    return "chat"


def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("classifier", classify)
    builder.add_node("metric_resolver", metric_resolver)
    builder.add_node("sql_generator", sql_generator)
    builder.add_node("query_tool", query_tool)
    builder.add_node("formatter", formatter)
    builder.add_node("intent_parser", intent_parser)
    builder.add_node("validate", validate)
    builder.add_node("tool_caller", tool_caller)
    builder.add_node("action_formatter", action_formatter)
    builder.add_node("chat_node", chat_node)

    builder.set_entry_point("classifier")
    builder.add_conditional_edges(
        "classifier",
        router,
        {
            "analytics": "metric_resolver",
            "action": "intent_parser",
            "chat": "chat_node",
        },
    )

    builder.add_edge("metric_resolver", "sql_generator")
    builder.add_edge("sql_generator", "query_tool")
    builder.add_edge("query_tool", "formatter")
    builder.add_edge("formatter", END)

    builder.add_edge("intent_parser", "validate")
    builder.add_edge("validate", "tool_caller")
    builder.add_edge("tool_caller", "action_formatter")
    builder.add_edge("action_formatter", END)

    builder.add_edge("chat_node", END)
    return builder.compile()


graph = build_graph()
app = FastAPI(title="Streaming Workflow API", version="1.0.0")


@app.get("/", response_model=HealthResponse)
@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse()


@app.post(
    "/v1/workflow",
    response_model=WorkflowResponse,
    responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
)
def invoke_workflow(payload: WorkflowRequest) -> WorkflowResponse:
    result = graph.invoke(
        {
            "user_query": payload.query,
            "intent": None,
            "parsed": None,
            "result": None,
        }
    )
    return WorkflowResponse(intent=result.get("intent"), result=result["result"])


@app.post(
    "/v1/workflow/stream",
    response_model=None,
    responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
)
async def stream_workflow(
    request: Request,
    payload: WorkflowRequest,
) -> StreamingResponse:
    async def event_stream():
        try:
            final_state: dict[str, object] | None = None
            yield _sse_event("start", {"query": payload.query})

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

                normalized_part = _normalize_stream_part(part)
                if normalized_part is None:
                    continue

                if normalized_part["type"] == "updates":
                    update_data = normalized_part["data"]
                    if isinstance(update_data, dict):
                        for node_name, node_update in update_data.items():
                            yield _sse_event(
                                "node",
                                {
                                    "node": node_name,
                                    "data": node_update,
                                },
                            )
                    continue

                if normalized_part["type"] == "values":
                    state = normalized_part["data"]
                    if isinstance(state, dict):
                        final_state = state

            if final_state is None:
                raise ValueError("Workflow completed without a final state.")

            yield _sse_event(
                "complete",
                {
                    "intent": final_state.get("intent"),
                    "parsed": final_state.get("parsed"),
                    "result": final_state.get("result"),
                },
            )
        except ValueError as exc:
            yield _sse_event("error", {"detail": str(exc)})
        except Exception as exc:
            logger.exception("Unhandled streaming workflow error", exc_info=exc)
            yield _sse_event("error", {"detail": "workflow request failed"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _normalize_stream_part(part: object) -> dict[str, object] | None:
    if isinstance(part, dict):
        return part
    if isinstance(part, tuple) and len(part) == 2 and isinstance(part[0], str):
        return {"type": part[0], "data": part[1]}
    return None


def _parse_json_object(raw_text: str) -> dict:
    candidate = raw_text.strip()
    if candidate.startswith("```"):
        lines = [line for line in candidate.splitlines() if not line.startswith("```")]
        candidate = "\n".join(lines).strip()
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object from the intent parser.")
    return parsed


def _stringify_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    return str(content)


def _sse_event(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def main() -> None:
    import uvicorn

    uvicorn.run(
        "streaming_workflow_api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("STREAMING_WORKFLOW_PORT", "8002")),
        reload=False,
    )


if __name__ == "__main__":
    main()
