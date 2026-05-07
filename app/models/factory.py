import os
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


class ProviderName(str, Enum):
    OPENROUTER = "openrouter"
    ZHIPU = "zhipu"
    GEMINI = "gemini"


@dataclass(frozen=True)
class ProviderConfig:
    name: ProviderName
    api_key: str
    base_url: str
    default_model: str


_PROVIDER_CONFIGS: dict[ProviderName, ProviderConfig] = {
    ProviderName.OPENROUTER: ProviderConfig(
        name=ProviderName.OPENROUTER,
        api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
        base_url="https://openrouter.ai/api/v1",
        default_model="openrouter/free",
    ),
    ProviderName.ZHIPU: ProviderConfig(
        name=ProviderName.ZHIPU,
        api_key=os.getenv("GLM_API_KEY", "").strip(),
        base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4-flash",
    ),
    ProviderName.GEMINI: ProviderConfig(
        name=ProviderName.GEMINI,
        api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        default_model="gemini-2.0-flash",
    ),
}


def _resolve_provider_and_model(
    provider: ProviderName | str | None,
    model: str | None,
) -> tuple[ProviderName, str]:
    if provider is None:
        resolved = ProviderName.OPENROUTER
    elif isinstance(provider, str):
        resolved = ProviderName(provider.lower())
    else:
        resolved = provider

    if resolved not in _PROVIDER_CONFIGS:
        raise ValueError(f"Unknown provider: {provider}. Available: {[p.value for p in ProviderName]}")

    cfg = _PROVIDER_CONFIGS[resolved]
    if not cfg.api_key:
        raise ValueError(f"API key not set for provider '{resolved.value}'. Check your .env file.")

    resolved_model = model if model else cfg.default_model
    return resolved, resolved_model


_llm_timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))


def get_llm(
    provider: ProviderName | str | None = None,
    model: str | None = None,
    temperature: float = 0,
    **kwargs,
) -> ChatOpenAI:
    resolved_provider, resolved_model = _resolve_provider_and_model(provider, model)
    cfg = _PROVIDER_CONFIGS[resolved_provider]
    return ChatOpenAI(
        model=resolved_model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        timeout=_llm_timeout,
        temperature=temperature,
        **kwargs,
    )


_DEFAULT_MODEL_OVERRIDES: dict[ProviderName, str] = {}


def get_default_provider() -> ProviderName:
    return ProviderName.OPENROUTER


def get_default_model(provider: ProviderName | str | None = None) -> str:
    prov = provider or ProviderName.OPENROUTER
    if isinstance(prov, str):
        prov = ProviderName(prov.lower())
    if prov not in _PROVIDER_CONFIGS:
        raise ValueError(f"Unknown provider: {prov}")
    if prov in _DEFAULT_MODEL_OVERRIDES:
        return _DEFAULT_MODEL_OVERRIDES[prov]
    return _PROVIDER_CONFIGS[prov].default_model


def set_default_model(provider: ProviderName | str, model: str) -> None:
    if isinstance(provider, str):
        provider = ProviderName(provider.lower())
    if provider not in _PROVIDER_CONFIGS:
        raise ValueError(f"Unknown provider: {provider}")
    cfg = _PROVIDER_CONFIGS[provider]
    if not cfg.api_key:
        raise ValueError(f"API key not set for provider '{provider.value}'")
    _DEFAULT_MODEL_OVERRIDES[provider] = model


def get_available_providers() -> list[dict]:
    available = []
    for name, cfg in _PROVIDER_CONFIGS.items():
        if cfg.api_key:
            effective_model = _DEFAULT_MODEL_OVERRIDES.get(name, cfg.default_model)
            available.append({
                "provider": name.value,
                "default_model": effective_model,
            })
    return available
