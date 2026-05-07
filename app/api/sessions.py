from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.repositories.sessions import delete_session, get_session, upsert_session
from app.models.factory import get_default_model, get_default_provider

router = APIRouter()


class SessionUpdate(BaseModel):
    provider: str | None = Field(default=None, min_length=1, max_length=32)
    model: str | None = Field(default=None, min_length=1, max_length=64)


class SessionResponse(BaseModel):
    session_id: str
    provider: str
    model: str


@router.get("/v1/sessions/{session_id}", response_model=SessionResponse)
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


@router.patch("/v1/sessions/{session_id}", response_model=SessionResponse)
async def update_session(session_id: str, payload: SessionUpdate) -> SessionResponse:
    existing = await get_session(session_id)
    provider = payload.provider or (existing.provider if existing else get_default_provider().value)
    model = payload.model or (existing.model if existing else get_default_model())
    session = await upsert_session(session_id, provider, model)
    return SessionResponse(
        session_id=session.session_id,
        provider=session.provider,
        model=session.model,
    )


@router.delete("/v1/sessions/{session_id}")
async def delete_session_endpoint(session_id: str) -> dict:
    await delete_session(session_id)
    return {"session_id": session_id, "deleted": True}
