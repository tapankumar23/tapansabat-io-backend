import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from stream_utils import normalize_stream_part, sse_event
from workflow_graph import GraphState, graph

logger = logging.getLogger(__name__)
router = APIRouter()


class WorkflowRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    workspace: str = Field(default="general", max_length=50)


class WorkflowResponse(BaseModel):
    result: str


class SchemaRequest(WorkflowRequest):
    """Same as WorkflowRequest but defaults workspace to the logistics schema."""
    workspace: str = Field(default="logistics-schema", max_length=50)


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _initial_state(payload: WorkflowRequest) -> dict:
    return {
        "user_query": payload.query,
        "workspace": payload.workspace,
        "intent": None,
        "schema_context": None,
        "parsed": None,
        "result": None,
    }


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


@router.post("/v1/workflow", response_model=WorkflowResponse)
async def chat_workflow(payload: WorkflowRequest) -> WorkflowResponse:
    result = await graph.ainvoke(_initial_state(payload))
    return WorkflowResponse(result=result["result"])


@router.post("/v1/workflow/stream", response_model=None)
async def stream_workflow(request: Request, payload: WorkflowRequest) -> StreamingResponse:
    async def event_stream():
        try:
            final_state: GraphState | None = None
            yield sse_event("start", {"query": payload.query})

            async for part in graph.astream(
                _initial_state(payload),
                stream_mode=["updates", "values"],
                version="v2",
            ):
                if await request.is_disconnected():
                    return

                normalized = normalize_stream_part(part)
                if normalized is None:
                    continue

                if normalized["type"] == "updates":
                    update_data = normalized["data"]
                    if isinstance(update_data, dict):
                        for node_name, node_update in update_data.items():
                            yield sse_event("node", {"node": node_name, "data": node_update})
                    continue

                if normalized["type"] == "values":
                    state = normalized["data"]
                    if isinstance(state, dict):
                        final_state = state

            if final_state is None:
                raise ValueError("Workflow completed without a final state.")

            yield sse_event("complete", {"result": final_state.get("result")})
        except ValueError as exc:
            yield sse_event("error", {"detail": str(exc), "request_id": _request_id(request)})
        except Exception:
            logger.exception("Unhandled streaming workflow error")
            yield sse_event("error", {"detail": "workflow request failed", "request_id": _request_id(request)})

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/v1/workflow/schema/stream", response_model=None)
async def stream_schema_workflow(request: Request, payload: SchemaRequest) -> StreamingResponse:
    async def event_stream():
        try:
            final_state: GraphState | None = None
            yield sse_event("start", {"query": payload.query})

            async for part in graph.astream(
                _initial_state(payload),
                stream_mode=["updates", "values"],
                version="v2",
            ):
                if await request.is_disconnected():
                    return

                normalized = normalize_stream_part(part)
                if normalized is None:
                    continue

                if normalized["type"] == "updates":
                    update_data = normalized["data"]
                    if isinstance(update_data, dict):
                        for node_name, node_update in update_data.items():
                            yield sse_event("node", {"node": node_name, "data": node_update})
                    continue

                if normalized["type"] == "values":
                    state = normalized["data"]
                    if isinstance(state, dict):
                        final_state = state

            if final_state is None:
                raise ValueError("Workflow completed without a final state.")

            yield sse_event("complete", {"result": final_state.get("result")})
        except ValueError as exc:
            yield sse_event("error", {"detail": str(exc), "request_id": _request_id(request)})
        except Exception:
            logger.exception("Unhandled schema workflow error")
            yield sse_event("error", {"detail": "workflow request failed", "request_id": _request_id(request)})

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


__all__ = ["router"]
