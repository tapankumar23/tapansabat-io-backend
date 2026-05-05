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

Conversation history is now persisted in Postgres via the LangGraph Postgres checkpointer. Set one of these env vars to your Supabase Postgres connection string:

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

Example request:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Hello","session_id":"demo-thread"}'
```

## Architecture

Four modules, each with a distinct responsibility:

- **[model_factory.py](model_factory.py)** — Provider abstraction. `PROVIDER_CONFIGS` maps `ProviderName` (`"glm"`, `"openrouter"`, `"gemini"`) to API base URLs and env var names. `get_provider_credentials(provider)` loads `.env` and returns `(ProviderConfig, api_key)` — it's also called directly by the `/readyz` endpoint to verify all keys are present at startup. `get_chat_model(provider, model_name)` wraps it to return a `ChatOpenAI` instance.

- **[chat_graph.py](chat_graph.py)** — LangGraph runtime factory. `initialize_chat_persistence()` opens an `AsyncPostgresSaver` using `LANGGRAPH_POSTGRES_URL`/`DATABASE_URL`/`SUPABASE_DB_URL`, optionally runs `setup()`, and keeps the saver alive for the app lifespan. `get_chat_app_async(provider, model_name)` builds and `@lru_cache`s a compiled `StateGraph` (up to 16 entries) so the same provider+model combination reuses the Postgres-backed checkpointer instead of in-memory state. `LazyChatApp` still defers graph construction to first use for script/notebook flows.

- **[chat_service.py](chat_service.py)** — Service layer. `ChatService.achat()` takes a message plus optional `session_id`/`provider`/`model_name`, auto-generates a UUID session if none is provided, invokes the correct cached graph, and normalizes the LLM response (handles both `str` and multi-part list content) into a `ChatResult`. `ChatService.chat()` delegates to the async path for script usage outside an event loop.

- **[chatbot_backend.py](chatbot_backend.py)** — FastAPI entry point. Exposes `/`, `/healthz`, `/readyz`, and `POST /v1/chat`. On startup it validates provider credentials, verifies the Postgres connection string, and initializes the shared `AsyncPostgresSaver`; on shutdown it closes the saver cleanly. Adds `x-request-id` propagation via middleware. `ValueError` → 400, unhandled exceptions → 502.

The default provider is `"openrouter"`. To use a different provider, pass `"provider"` in the `/v1/chat` JSON body.

## Key dependencies

| Package | Version constraint | Purpose |
|---|---|---|
| `langgraph` | ==1.1.10 | Graph-based agent orchestration |
| `langgraph-checkpoint-postgres` | >=2.0,<3.0 | Postgres-backed LangGraph checkpoint persistence |
| `langchain-openai` | ==1.2.1 | OpenAI-compatible LLM client |
| `langchain-core` | ==1.3.2 | Message types, base abstractions |
| `psycopg[binary,pool]` | >=3.2,<4.0 | PostgreSQL driver used by the checkpointer |
| `fastapi` | >=0.115,<1.0 | HTTP API framework |
| `uvicorn` | >=0.30,<1.0 | ASGI server |
| `python-dotenv` | ==1.2.2 | `.env` loading |
