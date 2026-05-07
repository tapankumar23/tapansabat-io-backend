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

Config is loaded from `.env` via `python-dotenv`. At least one LLM provider key is required.

Conversation history is persisted in Postgres via the LangGraph Postgres checkpointer. Set one of these env vars to your Supabase Postgres connection string:

- `LANGGRAPH_POSTGRES_URL`
- `DATABASE_URL`
- `SUPABASE_DB_URL`

For Supabase, use a Postgres URI that includes `sslmode=require`. First startup auto-creates LangGraph checkpoint tables unless `LANGGRAPH_POSTGRES_AUTO_SETUP=false`.

## Running the backend

```bash
python app/main.py    # port 8000
./run.sh              # port 8000 via uvicorn directly
```

## Configuration env vars

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | — | API key for OpenRouter (default provider) |
| `GLM_API_KEY` | — | API key for ZhipuAI (GLM models) |
| `GEMINI_API_KEY` | — | API key for Gemini |
| `LLM_TIMEOUT_SECONDS` | `30` | Timeout for every LLM API call |
| `MAX_MESSAGE_LENGTH` | `100` | Max characters in a chat message |
| `RATE_LIMIT_MAX_REQUESTS` | `1` | Requests allowed per window per client IP |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Sliding window duration in seconds |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `LANGGRAPH_POSTGRES_URL` / `DATABASE_URL` / `SUPABASE_DB_URL` | unset | Postgres connection string for chat checkpoints |
| `LANGGRAPH_POSTGRES_AUTO_SETUP` | `true` | Auto-create LangGraph checkpoint tables on startup |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `LOG_FORMAT` | `text` | Set to `json` for structured JSON logging |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Uvicorn bind address |
| `SQL_MAX_ROWS` | `500` | Hard cap on rows returned per SQL query |
| `SQL_POOL_SIZE` | `5` | Max connections in the SQL pool |

## Example requests

```bash
# Stateful chat (session mode, blocking)
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Hello","session_id":"demo-thread"}'

# Stateful chat with analytics mode (uses workflow graph, logistics-schema workspace)
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"How many shipments left BLR today?","mode":"analytics"}'

# Stateful chat (SSE streaming)
curl -X POST http://127.0.0.1:8000/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"Hello","session_id":"demo-thread"}'

# Stateless workflow (blocking)
curl -X POST http://127.0.0.1:8000/v1/workflow \
  -H 'Content-Type: application/json' \
  -d '{"query":"How many shipments left BLR today?","workspace":"logistics-schema"}'

# Stateless workflow (SSE streaming)
curl -X POST http://127.0.0.1:8000/v1/workflow/stream \
  -H 'Content-Type: application/json' \
  -d '{"query":"How many shipments left BLR today?","workspace":"logistics-schema"}'

# Schema-aware SQL workflow (SSE — defaults workspace to "logistics-schema")
curl -X POST http://127.0.0.1:8000/v1/workflow/schema/stream \
  -H 'Content-Type: application/json' \
  -d '{"query":"How many shipments left BLR today?"}'

# List available providers (only those with API keys set)
curl http://127.0.0.1:8000/v1/providers

# Update default model for a provider (in-memory, resets on restart)
curl -X PATCH http://127.0.0.1:8000/v1/providers/openrouter/default-model \
  -H 'Content-Type: application/json' \
  -d '{"model":"openai/gpt-4o"}'

# Fetch live OpenRouter model list
curl http://127.0.0.1:8000/v1/models

# Get/update/delete a chat session's stored provider+model
curl http://127.0.0.1:8000/v1/sessions/demo-thread
```

## Architecture

The code lives in the `app/` package with four sub-packages:

- **[app/main.py](app/main.py)** — FastAPI entry point. Lifespan opens/closes the Postgres checkpointer and SQL pool. Exception handlers: `RateLimitError` → 429 with `Retry-After`, `ValueError` → 400. Every response carries a server-generated `x-request-id`. Mounts routers from `api/chat.py`, `api/streaming.py`, `api/sessions.py`, `api/models.py`.

- **[app/models/factory.py](app/models/factory.py)** — Multi-provider LLM factory. `ProviderName` enum: `openrouter`, `zhipu`, `gemini`. `_PROVIDER_CONFIGS` maps each to its API key env var, base URL, and default model. `get_llm(provider, model, temperature)` returns a `ChatOpenAI` instance. `_DEFAULT_MODEL_OVERRIDES` is an in-memory dict — `set_default_model()` writes to it and `get_default_model()` reads from it; these resets on restart. Only providers with non-empty API keys appear in `get_available_providers()`.

- **[app/models/persistence.py](app/models/persistence.py)** — `AsyncPostgresSaver` singleton for LangGraph checkpointing. `initialize()` connects and optionally runs `setup()`; `close()` tears it down. `get_database_url()` reads the three DB URL env vars in order.

- **[app/utils/stream_utils.py](app/utils/stream_utils.py)** — `sse_event(event, data)` is the single SSE formatter. `normalize_stream_part` normalises the `(type, data)` tuple vs dict variants from LangGraph's multi-mode `astream`. `stringify_message_content` flattens string or list-of-content-blocks to plain text.

- **[app/utils/rate_limit.py](app/utils/rate_limit.py)** — Sliding-window rate limiter. `_MAX_REQUESTS` and `_WINDOW_SECONDS` are read once at import time. State is in-process only — not shared across workers.

- **[app/graphs/chat.py](app/graphs/chat.py)** — Stateful LangGraph graph factory. `_compiled_chat_app(provider, model)` is `lru_cache(maxsize=None)` keyed by provider+model. The graph has a single `chatbot` node; Postgres checkpointing gives it multi-turn memory keyed by `thread_id`.

- **[app/graphs/workflow.py](app/graphs/workflow.py)** — Stateless workflow graph. `_compiled_workflow_graph(provider, model)` is `lru_cache(maxsize=None)`. `classifier` routes to `analytics` or `general`; analytics runs `metric_resolver` → `sql_generator` → `query_tool` → `formatter`; general runs `chat_node`. `_WORKSPACE_TABLES` maps workspace names to DB tables — add a table name here to expose it to the SQL path. `run_workflow(query, workspace, provider, model)` is the public entry point.

- **[app/service.py](app/service.py)** — `ChatService` with `ChatMode` routing. `ChatMode.SESSION` uses the stateful chat graph with Postgres-persisted session history. `ChatMode.STATELESS` uses the workflow graph with `general` workspace. `ChatMode.ANALYTICS` uses the workflow graph with `logistics-schema` workspace. `achat()` returns `ChatResult | WorkflowResult`; `astream()` yields `ChatStreamEvent | WorkflowStreamEvent`. Accepts an injected `ChatAppFactory` protocol for testing.

- **[app/api/chat.py](app/api/chat.py)** — `/v1/chat` and `/v1/chat/stream`. `ChatRequest` includes `mode` (default `session`) and `workspace`. Rate-limited via `enforce_rate_limit` dependency.

- **[app/api/streaming.py](app/api/streaming.py)** — `/v1/workflow`, `/v1/workflow/stream`, `/v1/workflow/schema/stream`. `SchemaRequest` extends `WorkflowRequest` with a different `workspace` default (`"logistics-schema"`). All three call `run_workflow`.

- **[app/api/sessions.py](app/api/sessions.py)** — `GET/PATCH/DELETE /v1/sessions/{session_id}`. Sessions track `provider` and `model` per `session_id` in the `chat_sessions` Postgres table. `_ensure_sessions_table()` runs on every query (idempotent DDL).

- **[app/api/models.py](app/api/models.py)** — `GET /v1/model`, `GET /v1/models` (live OpenRouter API call via `httpx`), `GET /v1/providers`, `PATCH /v1/providers/{provider}/default-model`.

- **[app/api/sql.py](app/api/sql.py)** — `AsyncConnectionPool` (`psycopg_pool`) for SQL. `is_safe_sql` validates read-only queries (first token must be `SELECT` or `WITH`, blocks destructive patterns). `execute_sql` and `fetch_table_schema` use the pool; `fetch_table_schema` caches results in `_WORKSPACE_SCHEMA_CACHE` (no TTL). Pool uses `dict_row` — all rows are `dict`s.

## API documentation

`openapi.yaml` contains the full OpenAPI 3.1 specification for all endpoints. View it with:

```bash
npx @redocly/cli preview-docs openapi.yaml
```

## LangGraph Studio

`langgraph.json` is present but stale — it points to `./workflow_graph.py:graph` which no longer exists. The graph is now in `app/graphs/workflow.py` but is not exported as a module-level singleton. `langgraph dev` will not work without fixing `langgraph.json`.

## Development notes

No tests or linting configuration. `ChatService` uses `ChatAppFactory` (a `typing.Protocol`) for dependency injection — swappable in tests without touching the graph or Postgres. `app/api/sql.execute_sql` can be patched in isolation for workflow tests.

## Deployment caveat

The rate limiter (`_buckets` in `rate_limit.py`) and compiled graph caches (`lru_cache` in `graphs/chat.py` and `graphs/workflow.py`) are in-process. Running multiple uvicorn workers breaks per-IP rate limiting and causes each worker to build its own caches. Use a single worker or externalize rate limiting for multi-worker deployments.
