# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment setup

Uses a local `.venv` (Python 3.14). Activate before running anything:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

API keys are loaded from `.env` via `python-dotenv`. Required keys: `OPENROUTER_API_KEY`, `GLM_API_KEY`, `GEMINI_API_KEY`.

Conversation history is persisted in Postgres via the LangGraph Postgres checkpointer. Set one of these env vars to your Supabase Postgres connection string:

- `LANGGRAPH_POSTGRES_URL`
- `DATABASE_URL`
- `SUPABASE_DB_URL`

For Supabase, use a Postgres URI that includes `sslmode=require`. The first startup auto-creates the LangGraph checkpoint tables unless `LANGGRAPH_POSTGRES_AUTO_SETUP=false`.

## Running the backend

```bash
python chatbot_backend.py
```

This starts a FastAPI server on `http://0.0.0.0:8000` (overridable via `HOST`/`PORT` env vars).

## Configuration env vars

| Variable | Default | Purpose |
|---|---|---|
| `MAX_MESSAGE_LENGTH` | `100` | Max characters allowed in a chat message |
| `RATE_LIMIT_MAX_REQUESTS` | `1` | Requests allowed per window per client IP |
| `RATE_LIMIT_WINDOW_SECONDS` | `300` | Sliding window duration in seconds |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `LANGGRAPH_POSTGRES_URL` / `DATABASE_URL` / `SUPABASE_DB_URL` | unset | Supabase/Postgres connection string for persisted chat checkpoints |
| `LANGGRAPH_POSTGRES_AUTO_SETUP` | `true` | Auto-create LangGraph checkpoint tables on startup |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Uvicorn bind address |
| `STREAMING_WORKFLOW_PROVIDER` | `"gemini"` | LLM provider used by the workflow graph |
| `STREAMING_WORKFLOW_MODEL` | provider default | Model override for the workflow graph |

## Example requests

```bash
# Stateful chat (blocking)
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Hello","session_id":"demo-thread"}'

# Stateful chat (SSE streaming)
curl -X POST http://127.0.0.1:8000/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"Hello","session_id":"demo-thread"}'

# Stateless workflow (blocking)
curl -X POST http://127.0.0.1:8000/v1/workflow \
  -H 'Content-Type: application/json' \
  -d '{"query":"How many shipments left BLR today?"}'

# Stateless workflow (SSE streaming — emits node/complete/error events)
curl -X POST http://127.0.0.1:8000/v1/workflow/stream \
  -H 'Content-Type: application/json' \
  -d '{"query":"How many shipments left BLR today?"}'
```

## Architecture

Seven modules, each with a single responsibility:

- **[model_factory.py](model_factory.py)** — Provider abstraction. `PROVIDER_CONFIGS` maps `ProviderName` (`"glm"`, `"openrouter"`, `"gemini"`) to API base URLs and env var names. All three providers use the OpenAI-compatible `ChatOpenAI` client with different `base_url` values. `get_provider_credentials(provider)` returns `(ProviderConfig, api_key)`; `get_chat_model(provider, model_name)` wraps it. Calls `load_dotenv()` internally so it is self-contained when used from scripts or notebooks.

- **[chat_persistence.py](chat_persistence.py)** — Postgres checkpointer lifecycle. Owns the module-level `AsyncPostgresSaver` singleton. `initialize()` opens the connection and optionally runs `setup()`; `close()` tears it down. `get_checkpointer()` returns the live instance for graph compilation. Uses double-checked locking (`threading.Lock` wrapping `asyncio.Lock`) to safely initialize the singleton across concurrent async callers.

- **[chat_graph.py](chat_graph.py)** — Stateful LangGraph graph factory. `get_chat_app_async(provider, model_name)` ensures persistence is initialized then returns an `@lru_cache`d compiled `StateGraph` (up to 16 entries). The graph is a single `chatbot` node that appends to a `messages` list; Postgres checkpointing gives it multi-turn memory keyed by `thread_id`. `clear_cache()` is called on shutdown.

- **[chat_service.py](chat_service.py)** — Service layer. `ChatService` accepts an injected `ChatAppFactory` (a `Protocol`) so the graph backend is swappable without modifying the service. `achat()` normalizes input, invokes the graph, and returns a `ChatResult`. `astream()` yields `ChatStreamEvent` objects (`start` / `token` / `complete`) using LangGraph's `messages` + `values` dual stream mode. Two private helpers—`_normalize_stream_part` and `_stringify_message_content`—are also imported and reused by `streaming_workflow_api.py`.

- **[streaming_workflow_api.py](streaming_workflow_api.py)** — Stateless multi-step LangGraph workflow mounted as a FastAPI `APIRouter`. The graph classifies each query into `analytics`, `action`, or `chat` intent, then routes through dedicated node chains: analytics → `metric_resolver → sql_generator → query_tool → formatter`; action → `intent_parser → validate → tool_caller → action_formatter`; chat → `chat_node`. Unlike `chat_graph.py`, this graph has no checkpointer (stateless per request). The `POST /v1/workflow/stream` endpoint emits SSE `node` events for each graph step plus a final `complete` event.

- **[rate_limit.py](rate_limit.py)** — Sliding-window rate limiter. `enforce(request)` reads `RATE_LIMIT_MAX_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS` from env on each call and raises `RateLimitError` when the per-IP limit is exceeded. State is in-process only — not shared across workers.

- **[chatbot_backend.py](chatbot_backend.py)** — FastAPI entry point. Exposes `/`, `/healthz`, `/readyz`, `POST /v1/chat`, `POST /v1/chat/stream`, and mounts the workflow router (`/v1/workflow`, `/v1/workflow/stream`). `load_dotenv()` runs before local imports so env vars are available to all modules at import time. Lifespan validates credentials and initializes persistence on startup; closes and clears the graph cache on shutdown. `ValueError` → 400, `RateLimitError` → 429 with `Retry-After`, unhandled → 502. Every response carries an `x-request-id` header for tracing.

The default provider is `"gemini"`. To use a different provider, pass `"provider"` in the `/v1/chat` JSON body.

## Development notes

There are no tests or linting configuration in this repository. When adding them, note that `chat_service.py` uses `ChatAppFactory` (a `typing.Protocol`) for dependency injection, making `ChatService` testable with a mock factory without touching the graph or Postgres.

## Deployment caveat

The rate limiter (`_buckets` in `rate_limit.py`) and the compiled graph cache (`lru_cache` in `chat_graph.py`) are in-process. Running multiple uvicorn workers (e.g. `--workers 4`) breaks per-IP rate limiting and means each worker builds its own graph cache independently. Use a single worker or externalize rate limiting (e.g. via a reverse proxy) for multi-worker deployments.
