import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

ProviderName = Literal["glm", "openrouter", "gemini"]

_VALID_PROVIDERS = frozenset({"glm", "openrouter", "gemini"})
_raw_provider = os.getenv("DEFAULT_PROVIDER", "gemini").strip()
if _raw_provider not in _VALID_PROVIDERS:
    raise ValueError(
        f"Invalid DEFAULT_PROVIDER '{_raw_provider}'. Must be one of: {', '.join(sorted(_VALID_PROVIDERS))}"
    )
DEFAULT_PROVIDER: ProviderName = _raw_provider  # type: ignore[assignment]


@dataclass(frozen=True)
class ProviderConfig:
    env_var: str
    base_url: str
    default_model: str


PROVIDER_CONFIGS: dict[ProviderName, ProviderConfig] = {
    "glm": ProviderConfig(
        env_var="GLM_API_KEY",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        default_model="glm-4.5",
    ),
    "openrouter": ProviderConfig(
        env_var="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        default_model="openrouter/free",
    ),
    "gemini": ProviderConfig(
        env_var="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        default_model="gemini-flash-latest",
    ),
}


def get_chat_model(
    provider: ProviderName = DEFAULT_PROVIDER,
    model_name: str | None = None,
    **kwargs,
) -> ChatOpenAI:
    config, api_key = get_provider_credentials(provider=provider)
    kwargs.setdefault("timeout", float(os.getenv("LLM_TIMEOUT_SECONDS", "30")))
    return ChatOpenAI(
        model=model_name or config.default_model,
        api_key=api_key,
        base_url=config.base_url,
        **kwargs,
    )


def get_provider_credentials(
    provider: ProviderName = DEFAULT_PROVIDER,
) -> tuple[ProviderConfig, str]:
    config = PROVIDER_CONFIGS[provider]
    api_key = os.getenv(config.env_var)
    if not api_key:
        raise ValueError(f"{config.env_var} is not set. Check your .env file.")
    return config, api_key
