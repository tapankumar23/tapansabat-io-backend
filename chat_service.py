import asyncio
from dataclasses import dataclass
from uuid import uuid4

from langchain_core.messages import HumanMessage

from chat_graph import get_chat_app_async
from model_factory import ProviderName


@dataclass(frozen=True)
class ChatResult:
    session_id: str
    reply: str
    provider: ProviderName
    model_name: str | None = None


class ChatService:
    def chat(
        self,
        *,
        message: str,
        session_id: str | None = None,
        provider: ProviderName = "openrouter",
        model_name: str | None = None,
    ) -> ChatResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.achat(
                    message=message,
                    session_id=session_id,
                    provider=provider,
                    model_name=model_name,
                )
            )
        raise RuntimeError("Use `await ChatService.achat(...)` when running inside an event loop.")

    async def achat(
        self,
        *,
        message: str,
        session_id: str | None = None,
        provider: ProviderName = "openrouter",
        model_name: str | None = None,
    ) -> ChatResult:
        cleaned_message, active_session_id = _normalize_chat_input(
            message=message,
            session_id=session_id,
        )
        app = await get_chat_app_async(provider=provider, model_name=model_name)
        result = await app.ainvoke(
            {"messages": [HumanMessage(content=cleaned_message)]},
            config={"configurable": {"thread_id": active_session_id}},
        )

        return ChatResult(
            session_id=active_session_id,
            reply=_stringify_message_content(result["messages"][-1].content),
            provider=provider,
            model_name=model_name,
        )


def _normalize_chat_input(*, message: str, session_id: str | None) -> tuple[str, str]:
    cleaned_message = message.strip()
    if not cleaned_message:
        raise ValueError("message must not be empty")

    active_session_id = (session_id or "").strip() or uuid4().hex
    return cleaned_message, active_session_id


def _stringify_message_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        if text_parts:
            return "\n".join(text_parts)
    return str(content)
