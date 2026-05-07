import httpx
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.models.factory import (
    get_available_providers,
    get_default_model,
    get_default_provider,
    set_default_model,
)

router = APIRouter()


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


class DefaultModelUpdate(BaseModel):
    model: str | None = Field(default=None, min_length=1, max_length=64)


class OpenRouterModel(BaseModel):
    id: str
    name: str
    created: int
    description: str


class ModelsResponse(BaseModel):
    models: list[OpenRouterModel]


@router.get("/v1/model", response_model=ModelResponse)
def get_model() -> ModelResponse:
    return ModelResponse(provider=get_default_provider().value, model=get_default_model())


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://openrouter.ai/api/v1/models", timeout=15.0)
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


@router.get("/v1/providers", response_model=ProvidersResponse)
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


@router.patch("/v1/providers/{provider}/default-model", response_model=ProviderInfo)
def update_provider_default_model(
    provider: str,
    model: str | None = None,
) -> ProviderInfo:
    resolved_model = model if model else get_default_model(provider)
    set_default_model(provider, resolved_model)
    return ProviderInfo(provider=provider, default_model=resolved_model)
