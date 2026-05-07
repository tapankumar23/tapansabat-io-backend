import logging
import os
from typing import Annotated, Literal, Union

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.service import ChatMode, ChatService, WorkflowService
from app.utils.rate_limit import enforce as enforce_rate_limit
from app.utils.stream_utils import sse_event

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "100"))

_WORKFLOW_WORKSPACES: dict[ChatMode, str] = {
    ChatMode.ANALYTICS: "logistics-schema",
    ChatMode.STATELESS: "general",
}


class ErrorResponse(BaseModel):
    detail: str
    request_id: str


class SessionChatRequest(BaseModel):
    mode: Literal[ChatMode.SESSION] = ChatMode.SESSION
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    provider: str | None = Field(default=None, min_length=1, max_length=32)
    model: str | None = Field(default=None, min_length=1, max_length=64)


class WorkflowChatRequest(BaseModel):
    mode: Literal[ChatMode.STATELESS, ChatMode.ANALYTICS]
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    provider: str | None = Field(default=None, min_length=1, max_length=32)
    model: str | None = Field(default=None, min_length=1, max_length=64)
    workspace: str = Field(default="general", max_length=50)


ChatRequest = Annotated[
    Union[SessionChatRequest, WorkflowChatRequest],
    Field(discriminator="mode"),
]


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    provider: str
    model: str


class WorkflowResponse(BaseModel):
    result: str
    provider: str
    model: str


_session_service = ChatService()
_workflow_service = WorkflowService()


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


@router.post(
    "/v1/chat",
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
)
async def chat(
    payload: ChatRequest,
    _: None = Depends(enforce_rate_limit),
) -> ChatResponse | WorkflowResponse:
    if isinstance(payload, SessionChatRequest):
        result = await _session_service.achat(
            message=payload.message,
            session_id=payload.session_id,
            provider=payload.provider,
            model=payload.model,
        )
        return ChatResponse(
            session_id=result.session_id,
            reply=result.reply,
            provider=result.provider,
            model=result.model,
        )

    workspace = _WORKFLOW_WORKSPACES.get(payload.mode, "general")
    result = await _workflow_service.run(
        query=payload.message,
        workspace=workspace,
        provider=payload.provider,
        model=payload.model,
    )
    return WorkflowResponse(
        result=result.result,
        provider=result.provider,
        model=result.model,
    )


@router.post(
    "/v1/chat/stream",
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
)
async def chat_stream(
    request: Request,
    payload: ChatRequest,
    _: None = Depends(enforce_rate_limit),
) -> StreamingResponse:
    request_id_val = _request_id(request)

    async def event_stream():
        try:
            if isinstance(payload, SessionChatRequest):
                async for event in _session_service.astream(
                    message=payload.message,
                    session_id=payload.session_id,
                    provider=payload.provider,
                    model=payload.model,
                ):
                    if await request.is_disconnected():
                        return
                    payload_dict: dict = {
                        "session_id": event.session_id,
                        "provider": event.provider,
                        "model": event.model,
                    }
                    if event.delta:
                        payload_dict["delta"] = event.delta
                    if event.reply:
                        payload_dict["reply"] = event.reply
                    yield sse_event(event.event, payload_dict)
            else:
                workspace = _WORKFLOW_WORKSPACES.get(payload.mode, "general")
                async for event in _workflow_service.stream(
                    query=payload.message,
                    workspace=workspace,
                    provider=payload.provider,
                    model=payload.model,
                ):
                    if await request.is_disconnected():
                        return
                    payload_dict = {
                        "provider": event.provider,
                        "model": event.model,
                    }
                    if event.result:
                        payload_dict["result"] = event.result
                    if event.detail:
                        payload_dict["detail"] = event.detail
                    yield sse_event(event.event, payload_dict)

        except ValueError as exc:
            yield sse_event("error", {"detail": str(exc), "request_id": request_id_val})
        except Exception as exc:
            logger.exception("Unhandled chat stream error")
            yield sse_event("error", {"detail": str(exc), "request_id": request_id_val})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
