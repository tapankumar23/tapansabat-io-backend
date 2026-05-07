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
python chatbot_backend.py        # port 8000
./run.sh                         # port 8001 via uvicorn directly
```

## Configuration env vars

| Variable | Default | Purpose |
|---|---|---|
| `DEFAULT_PROVIDER` | `"gemini"` | Default LLM provider for chat endpoints |
| `STREAMING_WORKFLOW_PROVIDER` | `"gemini"` | LLM provider used by the workflow graph |
| `STREAMING_WORKFLOW_MODEL` | provider default | Model override for the workflow graph |
| `MAX_MESSAGE_LENGTH` | `100` | Max characters allowed in a chat message |
| `RATE_LIMIT_MAX_REQUESTS` | `1` | Requests allowed per window per client IP |
| `RATE_LIMIT_WINDOW_SECONDS` | `300` | Sliding window duration in seconds |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `LANGGRAPH_POSTGRES_URL` / `DATABASE_URL` / `SUPABASE_DB_URL` | unset | Supabase/Postgres connection string for persisted chat checkpoints |
| `LANGGRAPH_POSTGRES_AUTO_SETUP` | `true` | Auto-create LangGraph checkpoint tables on startup |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `LOG_FORMAT` | `text` | Set to `json` for structured JSON logging (e.g. in production) |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Uvicorn bind address |
| `LLM_TIMEOUT_SECONDS` | `30` | Timeout for every LLM API call |
| `SQL_MAX_ROWS` | `500` | Hard cap on rows returned per SQL query (a `LIMIT` is appended automatically) |
| `SQL_POOL_SIZE` | `5` | Max connections in the `workflow_sql` pool |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | unset | Enables Langfuse tracing; omit to run without tracing |
| `LANGFUSE_HOST` | Langfuse cloud | Override for self-hosted Langfuse (e.g. `http://localhost:3000`) |

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
  -d '{"query":"How many shipments left BLR today?","workspace":"general"}'

# Stateless workflow (SSE streaming — emits node/complete/error events)
curl -X POST http://127.0.0.1:8000/v1/workflow/stream \
  -H 'Content-Type: application/json' \
  -d '{"query":"How many shipments left BLR today?","workspace":"general"}'

# Schema-aware SQL workflow (SSE — same graph, defaults workspace to "logistics-schema")
curl -X POST http://127.0.0.1:8000/v1/workflow/schema/stream \
  -H 'Content-Type: application/json' \
  -d '{"query":"How many shipments left BLR today?","workspace":"logistics-schema"}'

# List configured providers and models
curl http://127.0.0.1:8000/v1/models
```

## Architecture

Eleven modules, each with a single responsibility:

- **[model_factory.py](model_factory.py)** — Provider abstraction. `PROVIDER_CONFIGS` maps `ProviderName` (`"glm"`, `"openrouter"`, `"gemini"`) to API base URLs and env var names. All three providers use the OpenAI-compatible `ChatOpenAI` client. `DEFAULT_PROVIDER` is validated at import time — an invalid value raises immediately. `get_chat_model` applies `LLM_TIMEOUT_SECONDS` (default 30s) to every LLM call via `kwargs.setdefault`.

- **[stream_utils.py](stream_utils.py)** — Shared streaming/content utilities. `sse_event(event, data)` is the single source of truth for SSE formatting (used by both `chatbot_backend.py` and `streaming_workflow_api.py`). `normalize_stream_part` normalises the `(type, data)` tuple vs dict variants from LangGraph's multi-mode `astream`. `stringify_message_content` flattens string or list-of-content-blocks into plain text.

- **[chat_persistence.py](chat_persistence.py)** — Postgres checkpointer lifecycle. Owns the module-level `AsyncPostgresSaver` singleton. `initialize()` opens the connection and optionally runs `setup()`; `close()` tears it down. Exports `get_database_url()` (reads the three DB URL env vars) which is also consumed by `workflow_sql.py`.

- **[chat_graph.py](chat_graph.py)** — Stateful LangGraph graph factory. `get_chat_app_async(provider, model_name)` returns an `@lru_cache`d compiled `StateGraph` (up to 16 entries). The graph is a single `chatbot` node that appends to a `messages` list; Postgres checkpointing gives it multi-turn memory keyed by `thread_id`.

- **[chat_service.py](chat_service.py)** — Stateful chat service layer. `ChatService` accepts an injected `ChatAppFactory` (a `Protocol`) so the graph backend is swappable. `achat()` returns a `ChatResult`; `astream()` yields `ChatStreamEvent` objects (`start` / `token` / `complete`).

- **[workflow_sql.py](workflow_sql.py)** — SQL safety, execution, and connection pooling. `is_safe_sql` validates read-only queries; `strip_sql_fences` removes LLM markdown fences; `_apply_row_limit` appends `LIMIT SQL_MAX_ROWS` if absent (prevents OOM from unbounded queries). `execute_sql` and `fetch_table_schema` share an `AsyncConnectionPool` (`psycopg_pool`) sized by `SQL_POOL_SIZE`. Call `open_pool()` / `close_pool()` in the app lifespan. `ping_db()` is used by `/readyz`.

- **[workflow_graph.py](workflow_graph.py)** — Stateless workflow graph. Defines `GraphState` (includes `workspace` and `schema_context`), the `get_llm()` singleton, and all node functions: `classify` (routes to `analytics` or `general`), `metric_resolver` (fetches live schema for the workspace via `fetch_table_schema`), `sql_generator`, `query_tool_async`, `formatter`, `chat_node`. `_WORKSPACE_TABLES` maps workspace names to the DB tables they may query — add a table name here to expose it, no column definitions needed. The compiled `graph` is a module-level singleton.

- **[streaming_workflow_api.py](streaming_workflow_api.py)** — FastAPI router for workflow endpoints. `SchemaRequest` extends `WorkflowRequest` with a different `workspace` default (`"logistics-schema"`). All three endpoints share `_initial_state()` and `_langfuse_config()` helpers and use the same `graph`. Every SSE error event includes a `request_id` for traceability. SSE formatting uses `sse_event` from `stream_utils`.

- **[langfuse_utils.py](langfuse_utils.py)** — Langfuse tracing integration. `get_langfuse_handler()` returns a per-request `CallbackHandler` when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set, or `None` otherwise. The handler is passed via `config={"callbacks": [...]}` at every graph and LLM call site so tracing is fully optional with no code changes.

- **[rate_limit.py](rate_limit.py)** — Sliding-window rate limiter. `_MAX_REQUESTS` and `_WINDOW_SECONDS` are read once at import time (changing them requires a restart). `_evict_stale()` runs hourly to remove dead IP buckets and prevent unbounded memory growth. State is in-process only — not shared across workers.

- **[chatbot_backend.py](chatbot_backend.py)** — FastAPI entry point. Exposes `/`, `/healthz`, `/readyz` (checks both API keys and DB connectivity via `ping_db`), `GET /v1/models`, `POST /v1/chat`, `POST /v1/chat/stream`, and mounts the workflow router. App lifespan opens/closes the SQL pool alongside Postgres persistence. `RateLimitError` → 429 with `Retry-After`, `ValueError` → 400, unhandled → 502. Every response carries a server-generated `x-request-id`. Supports JSON structured logging via `LOG_FORMAT=json`.

## LangGraph Studio

`langgraph.json` exposes the workflow `graph` for visual debugging:

```bash
source .venv/bin/activate
langgraph dev   # opens Studio UI, requires LANGSMITH_API_KEY in .env
```

## Development notes

There are no tests or linting configuration in this repository. `chat_service.py` uses `ChatAppFactory` (a `typing.Protocol`) for dependency injection, making `ChatService` testable with a mock factory without touching the graph or Postgres. `workflow_sql.execute_sql` can similarly be patched in isolation for workflow tests.

## Deployment caveat

The rate limiter (`_buckets` in `rate_limit.py`) and the compiled graph cache (`lru_cache` in `chat_graph.py` and `workflow_graph.py`) are in-process. Running multiple uvicorn workers breaks per-IP rate limiting and means each worker builds its own caches independently. Use a single worker or externalize rate limiting (e.g. via a reverse proxy) for multi-worker deployments.
