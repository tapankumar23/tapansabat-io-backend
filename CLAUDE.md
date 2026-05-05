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

- **[chat_graph.py](chat_graph.py)** — LangGraph runtime factory. `get_chat_app(provider, model_name)` builds and `@lru_cache`s a compiled `StateGraph` (up to 16 entries) so the same provider+model combination shares one `InMemorySaver` checkpointer, keeping conversation history alive across requests within a process. `LazyChatApp` is a shim that defers graph construction to first use (avoids loading credentials at import time).

- **[chat_service.py](chat_service.py)** — Service layer. `ChatService.chat()` takes a message plus optional `session_id`/`provider`/`model_name`, auto-generates a UUID session if none is provided, invokes the correct cached graph, and normalizes the LLM response (handles both `str` and multi-part list content) into a `ChatResult`.

- **[chatbot_backend.py](chatbot_backend.py)** — FastAPI entry point. Exposes `/`, `/healthz`, `/readyz`, and `POST /v1/chat`. Adds `x-request-id` propagation via middleware. `ValueError` → 400, unhandled exceptions → 502.

The default provider is `"openrouter"`. To use a different provider, pass `"provider"` in the `/v1/chat` JSON body.

## Key dependencies

| Package | Version constraint | Purpose |
|---|---|---|
| `langgraph` | ==1.1.10 | Graph-based agent orchestration |
| `langchain-openai` | ==1.2.1 | OpenAI-compatible LLM client |
| `langchain-core` | ==1.3.2 | Message types, base abstractions |
| `fastapi` | >=0.115,<1.0 | HTTP API framework |
| `uvicorn` | >=0.30,<1.0 | ASGI server |
| `python-dotenv` | ==1.2.2 | `.env` loading |
