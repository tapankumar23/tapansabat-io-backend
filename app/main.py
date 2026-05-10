import json
import logging
import os
from contextlib import asynccontextmanager
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()

from app.graphs.chat import clear_cache as clear_graph_cache
from app.models.persistence import close as close_persistence, initialize as initialize_persistence
from app.db.sql import close_pool as close_sql_pool, open_pool as open_sql_pool, ping_db
from app.db.indexer import index_all_workspaces
from app.repositories.sessions import ensure_sessions_table
from app.utils.rate_limit import RateLimitError
from app.utils.stream_utils import request_id as get_request_id
from app.api.chat import router as chat_router
from app.api.sessions import router as sessions_router
from app.api.models import router as models_router
from app.api.streaming import router as streaming_router


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
    await ensure_sessions_table()
    await index_all_workspaces()
    logger.info("Startup complete — Postgres ready, SQL pool open, schema embeddings indexed")
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


app = FastAPI(
    title="AI Chatbot Backend",
    version="1.0.0",
    description="""
Multi-provider AI chatbot API with stateful session support, stateless workflow mode,
and SQL analytics mode (controlled via the `mode` field on chat requests).

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
    ],
    tags=[
        {"name": "Health", "description": "Liveness and readiness probes"},
        {"name": "Models", "description": "Inspect the configured LLM"},
        {"name": "Chat", "description": "Stateful multi-turn chat with Postgres-persisted session history"},
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

app.include_router(chat_router)
app.include_router(sessions_router)
app.include_router(models_router)
app.include_router(streaming_router)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


@app.exception_handler(RateLimitError)
async def _handle_rate_limit(request: Request, exc: RateLimitError) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": str(exc.retry_after)},
        content={"detail": str(exc), "request_id": get_request_id(request)},
    )


@app.exception_handler(ValueError)
async def _handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "request_id": get_request_id(request)},
    )


@app.get("/healthz", response_model=HealthResponse, tags=["Health"])
async def healthz() -> HealthResponse:
    await ping_db()
    return HealthResponse()


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
