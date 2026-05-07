import httpx
import json
import logging
import os
from contextlib import asynccontextmanager
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

from app.graphs.chat import clear_cache as clear_graph_cache
from app.models.persistence import close as close_persistence, initialize as initialize_persistence
from app.api.sessions import delete_session, get_session, upsert_session, SessionUpdate
from app.api.sql import close_pool as close_sql_pool, open_pool as open_sql_pool, ping_db
from app.service import ChatService
from app.models.factory import (
    ProviderName,
    get_available_providers,
    get_default_model,
    get_default_provider,
    set_default_model,
)
from app.utils.rate_limit import RateLimitError, enforce as enforce_rate_limit
from app.utils.stream_utils import sse_event
from app.api.streaming import router as workflow_router

MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "100"))


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        obj: dict[str, object] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(obj)


def _configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO")
    fmt = os.getenv("LOG_FORMAT", "text").lower()
    handler = logging.StreamHandler()
    handler.setFormatter(
        _JsonFormatter()
        if fmt == "json"
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logging.basicConfig(level=level, handlers=[handler], force=True)


_configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await initialize_persistence()
    await open_sql_pool()
    logger.info("Startup complete — Postgres ready, SQL pool open")
    try:
        yield
    finally:
        await close_persistence()
        await close_sql_pool()
        clear_graph_cache()
        logger.info("Shutdown complete")


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "ai-chatbot-backend"


class ErrorResponse(BaseModel):
    detail: str
    request_id: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    provider: str | None = Field(default=None, min_length=1, max_length=32)
    model: str | None = Field(default=None, min_length=1, max_length=64)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    provider: str
    model: str


class ModelResponse(BaseModel):
    provider: str
    model: str


class ProviderInfo(BaseModel):
    provider: str
    default_model: str


class ProvidersResponse(BaseModel):
    default_provider: str
    default_model: str
    available_providers: list[ProviderInfo]


class SessionResponse(BaseModel):
    session_id: str
    provider: str
    model: str


class OpenRouterModel(BaseModel):
    id: str
    name: str
    created: int
    description: str


class ModelsResponse(BaseModel):
    models: list[OpenRouterModel]


service = ChatService()
app = FastAPI(
    title="AI Chatbot Backend",
    version="1.0.0",
    description="""
Multi-provider AI chatbot and stateless workflow API.

## Authentication
No authentication is required. API keys are server-side only.

## Rate Limiting
Chat endpoints are rate-limited per client IP. Exceeding the limit returns **429**
with a `Retry-After` header indicating when the client may retry.

## Request Tracing
Every response includes an `x-request-id` header (server-generated UUID hex).
Include this value when reporting errors.

## Server-Sent Events (SSE)
Streaming endpoints return `text/event-stream`. Each event has the shape:
```
event: <event-type>
data: <JSON object>
```
See individual endpoint descriptions for the event types they emit.
""",
    servers=[
        {"url": "http://localhost:8000", "description": "Local development"},
        {"url": "http://localhost:8001", "description": "Local development (run.sh)"},
    ],
    tags=[
        {"name": "Health", "description": "Liveness and readiness probes"},
        {"name": "Models", "description": "Inspect the configured LLM"},
        {"name": "Chat", "description": "Stateful multi-turn chat with Postgres-persisted session history"},
        {"name": "Workflow", "description": "Stateless intent-classified workflow — routes to SQL analytics or general chat"},
    ],
    lifespan=lifespan,
)

_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(workflow_router)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


@app.exception_handler(RateLimitError)
async def _handle_rate_limit(request: Request, exc: RateLimitError) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": str(exc.retry_after)},
        content={"detail": str(exc), "request_id": _request_id(request)},
    )


@app.exception_handler(ValueError)
async def _handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "request_id": _request_id(request)},
    )


@app.get("/healthz", response_model=HealthResponse, tags=["Health"])
async def healthz() -> HealthResponse:
    await ping_db()
    return HealthResponse()


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


@app.get("/v1/model", response_model=ModelResponse, tags=["Models"])
def get_model() -> ModelResponse:
    return ModelResponse(provider=get_default_provider().value, model=get_default_model())


@app.get("/v1/models", response_model=ModelsResponse, tags=["Models"])
async def list_models() -> ModelsResponse:
    async with httpx.AsyncClient() as client:
        resp = await client.get(OPENROUTER_MODELS_URL, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
    raw_models: list[dict] = data.get("data", [])
    models = [
        OpenRouterModel(
            id=m["id"],
            name=m.get("name", m["id"]),
            created=m.get("created", 0),
            description=m.get("description", ""),
        )
        for m in raw_models
    ]
    return ModelsResponse(models=models)


@app.get("/v1/providers", response_model=ProvidersResponse, tags=["Models"])
def list_providers() -> ProvidersResponse:
    default_provider = get_default_provider()
    return ProvidersResponse(
        default_provider=default_provider.value,
        default_model=get_default_model(),
        available_providers=[
            ProviderInfo(provider=p["provider"], default_model=p["default_model"])
            for p in get_available_providers()
        ],
    )


class DefaultModelUpdate(BaseModel):
    model: str = Field(min_length=1, max_length=64)


@app.patch("/v1/providers/{provider}/default-model", response_model=ProviderInfo, tags=["Models"])
def update_provider_default_model(provider: str, payload: DefaultModelUpdate) -> ProviderInfo:
    set_default_model(provider, payload.model)
    return ProviderInfo(provider=provider, default_model=payload.model)


@app.get("/v1/sessions/{session_id}", response_model=SessionResponse, tags=["Sessions"])
async def get_session_info(session_id: str) -> SessionResponse:
    session = await get_session(session_id)
    if session is None:
        return SessionResponse(
            session_id=session_id,
            provider=get_default_provider().value,
            model=get_default_model(),
        )
    return SessionResponse(
        session_id=session.session_id,
        provider=session.provider,
        model=session.model,
    )


@app.patch("/v1/sessions/{session_id}", response_model=SessionResponse, tags=["Sessions"])
async def update_session(
    session_id: str,
    payload: SessionUpdate,
) -> SessionResponse:
    existing = await get_session(session_id)
    provider = payload.provider or (existing.provider if existing else get_default_provider().value)
    model = payload.model or (existing.model if existing else get_default_model())
    session = await upsert_session(session_id, provider, model)
    return SessionResponse(
        session_id=session.session_id,
        provider=session.provider,
        model=session.model,
    )


@app.delete("/v1/sessions/{session_id}", response_model=None, tags=["Sessions"])
async def delete_session_endpoint(session_id: str) -> dict:
    await delete_session(session_id)
    return {"session_id": session_id, "deleted": True}


@app.post(
    "/v1/chat",
    response_model=ChatResponse,
    tags=["Chat"],
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
        model=payload.model,
    )
    return ChatResponse(
        session_id=result.session_id,
        reply=result.reply,
        provider=result.provider,
        model=result.model,
    )


@app.post(
    "/v1/chat/stream",
    response_model=None,
    tags=["Chat"],
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
    request_id = _request_id(request)

    async def event_stream():
        try:
            async for event in service.astream(
                message=payload.message,
                session_id=payload.session_id,
                provider=payload.provider,
                model=payload.model,
            ):
                if await request.is_disconnected():
                    return

                event_payload: dict[str, object] = {
                    "session_id": event.session_id,
                    "provider": event.provider,
                    "model": event.model,
                }
                if event.delta:
                    event_payload["delta"] = event.delta
                if event.reply:
                    event_payload["reply"] = event.reply

                yield sse_event(event.event, event_payload)
        except ValueError as exc:
            yield sse_event("error", {"detail": str(exc), "request_id": request_id})
        except Exception:
            logger.exception("Unhandled chatbot stream error")
            yield sse_event("error", {"detail": "chat request failed", "request_id": request_id})

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
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
