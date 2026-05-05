import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


ProviderName = Literal["glm", "openrouter", "gemini"]


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
    provider: ProviderName = "openrouter",
    model_name: str | None = None,
    dotenv_path: str = ".env",
    **kwargs,
) -> ChatOpenAI:
    config, api_key = get_provider_credentials(provider=provider, dotenv_path=dotenv_path)

    return ChatOpenAI(
        model=model_name or config.default_model,
        api_key=api_key,
        base_url=config.base_url,
        **kwargs,
    )


def get_provider_credentials(
    provider: ProviderName = "openrouter",
    dotenv_path: str = ".env",
) -> tuple[ProviderConfig, str]:
    load_dotenv(dotenv_path=dotenv_path)

    config = PROVIDER_CONFIGS[provider]
    api_key = os.getenv(config.env_var)
    if not api_key:
        raise ValueError(f"{config.env_var} is not set. Check your .env file.")
    return config, api_key


def get_glm_model(
    model_name: str = "glm-4.5",
    dotenv_path: str = ".env",
    **kwargs,
) -> ChatOpenAI:
    return get_chat_model(
        provider="glm",
        model_name=model_name,
        dotenv_path=dotenv_path,
        **kwargs,
    )


def get_openrouter_model(
    model_name: str = "openrouter/free",
    dotenv_path: str = ".env",
    **kwargs,
) -> ChatOpenAI:
    return get_chat_model(
        provider="openrouter",
        model_name=model_name,
        dotenv_path=dotenv_path,
        **kwargs,
    )


def get_gemini_model(
    model_name: str = "gemini-flash-latest",
    dotenv_path: str = ".env",
    **kwargs,
) -> ChatOpenAI:
    return get_chat_model(
        provider="gemini",
        model_name=model_name,
        dotenv_path=dotenv_path,
        **kwargs,
    )
