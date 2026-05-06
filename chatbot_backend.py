import logging
import os
import json
from contextlib import asynccontextmanager
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

from chat_graph import clear_cache as clear_graph_cache  # noqa: E402
from chat_persistence import close as close_persistence  # noqa: E402
from chat_persistence import get_database_url  # noqa: E402
from chat_persistence import initialize as initialize_persistence  # noqa: E402
from chat_service import ChatService  # noqa: E402
from model_factory import DEFAULT_PROVIDER, PROVIDER_CONFIGS, ProviderName, get_provider_credentials  # noqa: E402
from rate_limit import RateLimitError, enforce as enforce_rate_limit  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "100"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    for provider in PROVIDER_CONFIGS:
        get_provider_credentials(provider)
    get_database_url()
    await initialize_persistence()
    logger.info("All provider credentials verified")
    logger.info("Postgres chat persistence initialized")
    try:
        yield
    finally:
        await close_persistence()
        clear_graph_cache()


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "ai-chatbot-backend"


class ErrorResponse(BaseModel):
    detail: str
    request_id: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    provider: ProviderName = DEFAULT_PROVIDER
    model_name: str | None = Field(default=None, max_length=128)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    provider: ProviderName
    model_name: str | None = None


service = ChatService()
app = FastAPI(title="AI Chatbot Backend", version="1.0.0", lifespan=lifespan)

_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _sse_event(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.exception_handler(RateLimitError)
async def handle_rate_limit(request: Request, exc: RateLimitError) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": str(exc.retry_after)},
        content=ErrorResponse(detail=str(exc), request_id=_request_id(request)).model_dump(),
    )


@app.exception_handler(ValueError)
async def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(detail=str(exc), request_id=_request_id(request)).model_dump(),
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled chatbot backend error", exc_info=exc)
    return JSONResponse(
        status_code=502,
        content=ErrorResponse(detail="chat request failed", request_id=_request_id(request)).model_dump(),
    )


@app.get("/", response_model=HealthResponse)
@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse()


@app.get("/readyz", response_model=HealthResponse)
def readyz() -> HealthResponse:
    for provider in PROVIDER_CONFIGS:
        get_provider_credentials(provider)
    get_database_url()
    return HealthResponse()


@app.post(
    "/v1/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
)
async def chat(
    payload: ChatRequest,
    _: None = Depends(enforce_rate_limit),
) -> ChatResponse:
    result = await service.achat(
        message=payload.message,
        session_id=payload.session_id,
        provider=payload.provider,
        model_name=payload.model_name,
    )

    return ChatResponse(
        session_id=result.session_id,
        reply=result.reply,
        provider=result.provider,
        model_name=result.model_name,
    )


@app.post(
    "/v1/chat/stream",
    response_model=None,
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
    async def event_stream():
        try:
            async for event in service.astream(
                message=payload.message,
                session_id=payload.session_id,
                provider=payload.provider,
                model_name=payload.model_name,
            ):
                if await request.is_disconnected():
                    return

                event_payload = {
                    "session_id": event.session_id,
                    "provider": event.provider,
                    "model_name": event.model_name,
                }
                if event.delta:
                    event_payload["delta"] = event.delta
                if event.reply:
                    event_payload["reply"] = event.reply

                yield _sse_event(event.event, event_payload)
        except ValueError as exc:
            yield _sse_event(
                "error",
                {"detail": str(exc), "request_id": _request_id(request)},
            )
        except Exception as exc:
            logger.exception("Unhandled chatbot stream error", exc_info=exc)
            yield _sse_event(
                "error",
                {"detail": "chat request failed", "request_id": _request_id(request)},
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def main() -> None:
    import uvicorn

    uvicorn.run(
        "chatbot_backend:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
