import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from chat_graph import (
    close_chat_persistence,
    get_checkpoint_database_url,
    initialize_chat_persistence,
)
from chat_service import ChatService
from model_factory import PROVIDER_CONFIGS, ProviderName, get_provider_credentials

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "100"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "1"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "300"))

# sliding-window counters keyed by client IP
_rate_buckets: dict[str, deque] = defaultdict(deque)


class RateLimitError(Exception):
    def __init__(self, retry_after: int) -> None:
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after}s.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    for provider in PROVIDER_CONFIGS:
        get_provider_credentials(provider)
    get_checkpoint_database_url()
    await initialize_chat_persistence()
    logger.info("All provider credentials verified")
    logger.info("Postgres chat persistence initialized")
    try:
        yield
    finally:
        await close_chat_persistence()


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "ai-chatbot-backend"


class ErrorResponse(BaseModel):
    detail: str
    request_id: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    provider: ProviderName = "openrouter"
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


@app.exception_handler(RateLimitError)
async def handle_rate_limit(request: Request, exc: RateLimitError) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": str(exc.retry_after)},
        content=ErrorResponse(
            detail=str(exc),
            request_id=getattr(request.state, "request_id", "unknown"),
        ).model_dump(),
    )


@app.exception_handler(ValueError)
async def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            detail=str(exc),
            request_id=getattr(request.state, "request_id", "unknown"),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled chatbot backend error", exc_info=exc)
    return JSONResponse(
        status_code=502,
        content=ErrorResponse(
            detail="chat request failed",
            request_id=getattr(request.state, "request_id", "unknown"),
        ).model_dump(),
    )


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(request: Request) -> None:
    ip = _client_ip(request)
    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    bucket = _rate_buckets[ip]

    while bucket and bucket[0] < window_start:
        bucket.popleft()

    if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
        retry_after = int(bucket[0] - window_start) + 1
        raise RateLimitError(retry_after)

    bucket.append(now)


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse()


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse()


@app.get("/readyz", response_model=HealthResponse)
async def readyz() -> HealthResponse:
    for provider in PROVIDER_CONFIGS:
        get_provider_credentials(provider)
    get_checkpoint_database_url()
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
    _: None = Depends(_enforce_rate_limit),
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
