import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.service import WorkflowService
from app.utils.rate_limit import enforce as enforce_rate_limit
from app.utils.stream_utils import request_id as get_request_id, sse_event

logger = logging.getLogger(__name__)
router = APIRouter()

_workflow_service = WorkflowService()


class WorkflowRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    workspace: str = Field(default="general", max_length=50)
    provider: str | None = Field(default=None, max_length=32)
    model: str | None = Field(default=None, max_length=64)


class WorkflowResponse(BaseModel):
    result: str
    provider: str
    model: str


class SchemaRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    workspace: str = Field(default="logistics-schema", max_length=50)
    provider: str | None = Field(default=None, max_length=32)
    model: str | None = Field(default=None, max_length=64)


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.post("/v1/workflow", response_model=WorkflowResponse, tags=["Workflow"])
async def chat_workflow(payload: WorkflowRequest, _: None = Depends(enforce_rate_limit)) -> WorkflowResponse:
    result = await _workflow_service.run(
        query=payload.query,
        workspace=payload.workspace,
        provider=payload.provider,
        model=payload.model,
    )
    return WorkflowResponse(result=result.result, provider=result.provider, model=result.model)


@router.post("/v1/workflow/stream", response_model=None, tags=["Workflow"])
async def stream_workflow(request: Request, payload: WorkflowRequest, _: None = Depends(enforce_rate_limit)) -> StreamingResponse:
    async def event_stream():
        try:
            async for event in _workflow_service.stream(
                query=payload.query,
                workspace=payload.workspace,
                provider=payload.provider,
                model=payload.model,
            ):
                yield sse_event(event.event, {
                    "provider": event.provider,
                    "model": event.model,
                    **({"result": event.result} if event.result else {}),
                    **({"detail": event.detail} if event.detail else {}),
                })
        except ValueError as exc:
            yield sse_event("error", {"detail": str(exc), "request_id": get_request_id(request)})
        except Exception:
            logger.exception("Unhandled streaming workflow error")
            yield sse_event("error", {"detail": "workflow request failed", "request_id": get_request_id(request)})

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/v1/workflow/schema/stream", response_model=None, tags=["Workflow"])
async def stream_schema_workflow(request: Request, payload: SchemaRequest, _: None = Depends(enforce_rate_limit)) -> StreamingResponse:
    async def event_stream():
        try:
            async for event in _workflow_service.stream(
                query=payload.query,
                workspace=payload.workspace,
                provider=payload.provider,
                model=payload.model,
            ):
                yield sse_event(event.event, {
                    "provider": event.provider,
                    "model": event.model,
                    **({"result": event.result} if event.result else {}),
                    **({"detail": event.detail} if event.detail else {}),
                })
        except ValueError as exc:
            yield sse_event("error", {"detail": str(exc), "request_id": get_request_id(request)})
        except Exception:
            logger.exception("Unhandled schema workflow error")
            yield sse_event("error", {"detail": "workflow request failed", "request_id": get_request_id(request)})

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


__all__ = ["router"]
