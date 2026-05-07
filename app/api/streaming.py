import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.utils.stream_utils import normalize_stream_part, sse_event
from app.graphs.workflow import run_schema_workflow, run_workflow

logger = logging.getLogger(__name__)
router = APIRouter()


class WorkflowRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    workspace: str = Field(default="general", max_length=50)
    provider: str | None = Field(default=None, max_length=32)
    model: str | None = Field(default=None, max_length=64)


class WorkflowResponse(BaseModel):
    result: str
    provider: str
    model: str


class SchemaRequest(WorkflowRequest):
    workspace: str = Field(default="logistics-schema", max_length=50)


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


@router.post("/v1/workflow", response_model=WorkflowResponse, tags=["Workflow"])
async def chat_workflow(payload: WorkflowRequest) -> WorkflowResponse:
    result = await run_workflow(
        query=payload.query,
        workspace=payload.workspace,
        provider=payload.provider,
        model=payload.model,
    )
    return WorkflowResponse(
        result=result["result"],
        provider=result["provider"],
        model=result["model"],
    )


@router.post("/v1/workflow/stream", response_model=None, tags=["Workflow"])
async def stream_workflow(request: Request, payload: WorkflowRequest) -> StreamingResponse:
    async def event_stream():
        try:
            final_state = await run_workflow(
                query=payload.query,
                workspace=payload.workspace,
                provider=payload.provider,
                model=payload.model,
            )
            yield sse_event("complete", {
                "result": final_state.get("result"),
                "provider": final_state.get("provider"),
                "model": final_state.get("model"),
            })
        except ValueError as exc:
            yield sse_event("error", {"detail": str(exc), "request_id": _request_id(request)})
        except Exception:
            logger.exception("Unhandled streaming workflow error")
            yield sse_event("error", {"detail": "workflow request failed", "request_id": _request_id(request)})

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/v1/workflow/schema/stream", response_model=None, tags=["Workflow"])
async def stream_schema_workflow(request: Request, payload: SchemaRequest) -> StreamingResponse:
    async def event_stream():
        try:
            yield sse_event("start", {"query": payload.query})
            final_state = await run_schema_workflow(
                query=payload.query,
                workspace=payload.workspace,
                provider=payload.provider,
                model=payload.model,
            )
            yield sse_event("complete", {
                "result": final_state.get("result"),
                "provider": final_state.get("provider"),
                "model": final_state.get("model"),
            })
        except ValueError as exc:
            yield sse_event("error", {"detail": str(exc), "request_id": _request_id(request)})
        except Exception:
            logger.exception("Unhandled schema workflow error")
            yield sse_event("error", {"detail": "workflow request failed", "request_id": _request_id(request)})

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


__all__ = ["router"]
