# API Sequence Diagram

<div style="width:100%;overflow-x:auto">

```mermaid
sequenceDiagram
    participant C as Client
    participant API as FastAPI
    participant RL as RateLimit
    participant SVC as Service Layer
    participant DB as Postgres
    participant LLM as LLM Provider
    participant OR as OpenRouter API

    %% ── POST /v1/chat (session mode, blocking) ──
    Note over C,LLM: POST /v1/chat  {mode: "session"}
    C->>API: POST /v1/chat
    API->>RL: enforce_rate_limit
    RL-->>API: ok
    API->>SVC: ChatService.achat()
    SVC->>DB: get_session (chat_sessions)
    DB-->>SVC: stored provider/model
    SVC->>DB: upsert_session
    SVC->>DB: ainvoke graph (checkpoint read)
    DB-->>SVC: prior messages
    SVC->>LLM: ainvoke
    LLM-->>SVC: AIMessage
    SVC->>DB: checkpoint write (checkpoint_blobs)
    SVC-->>API: ChatResult
    API-->>C: {session_id, reply, provider, model}

    %% ── POST /v1/chat/stream (session mode, SSE) ──
    Note over C,LLM: POST /v1/chat/stream  {mode: "session"}
    C->>API: POST /v1/chat/stream
    API->>RL: enforce_rate_limit
    RL-->>API: ok
    API-->>C: SSE: event:start
    API->>SVC: ChatService.astream()
    SVC->>DB: get_session / upsert_session
    loop per token chunk
        SVC->>LLM: stream token
        LLM-->>SVC: AIMessageChunk
        SVC-->>API: ChatStreamEvent(token)
        API-->>C: SSE: event:token {delta}
    end
    SVC->>DB: checkpoint write
    API-->>C: SSE: event:complete {reply}

    %% ── POST /v1/chat (stateless/analytics mode) ──
    Note over C,LLM: POST /v1/chat  {mode: "stateless"|"analytics"}
    C->>API: POST /v1/chat
    API->>RL: enforce_rate_limit
    API->>SVC: WorkflowService.run()
    SVC->>LLM: classify intent
    LLM-->>SVC: "analytics" | "general"
    alt analytics
        SVC->>DB: fetch_table_schema
        DB-->>SVC: schema DDL
        SVC->>LLM: generate SQL
        LLM-->>SVC: SQL query
        SVC->>DB: execute_sql
        DB-->>SVC: rows
        SVC->>LLM: format result
        LLM-->>SVC: formatted text
    else general
        SVC->>LLM: chat_node
        LLM-->>SVC: reply
    end
    API-->>C: {result, provider, model}

    %% ── POST /v1/workflow (blocking) ──
    Note over C,LLM: POST /v1/workflow
    C->>API: POST /v1/workflow
    API->>RL: enforce_rate_limit
    API->>SVC: WorkflowService.run()
    Note right of SVC: same analytics/general branch as above
    SVC-->>API: WorkflowResult
    API-->>C: {result, provider, model}

    %% ── POST /v1/workflow/stream (SSE) ──
    Note over C,LLM: POST /v1/workflow/stream  (and /schema/stream)
    C->>API: POST /v1/workflow/stream
    API->>RL: enforce_rate_limit
    API-->>C: SSE: event:start
    API->>SVC: WorkflowService.stream()
    Note right of SVC: runs full workflow graph
    SVC-->>API: WorkflowStreamEvent(complete)
    API-->>C: SSE: event:complete {result}

    %% ── Sessions ──
    Note over C,DB: Session management
    C->>API: GET /v1/sessions/{id}
    API->>DB: get_session
    DB-->>API: provider, model (or defaults)
    API-->>C: {session_id, provider, model}

    C->>API: PATCH /v1/sessions/{id}
    API->>DB: upsert_session
    DB-->>API: updated row
    API-->>C: {session_id, provider, model}

    C->>API: DELETE /v1/sessions/{id}
    API->>DB: delete_session
    API-->>C: {session_id, deleted: true}

    %% ── Models / Providers ──
    Note over C,OR: Model & provider management
    C->>API: GET /v1/model
    API-->>C: {provider, model}  (in-memory defaults)

    C->>API: GET /v1/models
    API->>OR: GET openrouter.ai/api/v1/models
    OR-->>API: model list
    API-->>C: {models: [...]}

    C->>API: GET /v1/providers
    API-->>C: {default_provider, default_model, available_providers}

    C->>API: PATCH /v1/providers/{provider}/default-model
    API-->>C: {provider, default_model}  (in-memory update)
```

</div>

## Endpoint Reference

| Endpoint | Persistence | Streaming | Graph |
|---|---|---|---|
| `POST /v1/chat` (session) | Postgres checkpoint + chat_sessions | No | chat.py |
| `POST /v1/chat/stream` (session) | Postgres checkpoint + chat_sessions | SSE tokens | chat.py |
| `POST /v1/chat` (stateless/analytics) | None | No | workflow.py |
| `POST /v1/workflow` | None | No | workflow.py |
| `POST /v1/workflow/stream` | None | SSE start+complete | workflow.py |
| `POST /v1/workflow/schema/stream` | None | SSE start+complete | workflow.py (logistics-schema default) |
| `GET/PATCH/DELETE /v1/sessions/{id}` | chat_sessions table | — | — |
| `GET /v1/model`, `GET /v1/providers` | In-memory | — | — |
| `PATCH /v1/providers/{p}/default-model` | In-memory (resets on restart) | — | — |
| `GET /v1/models` | Proxies OpenRouter API | — | — |
