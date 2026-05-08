from functools import lru_cache
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.models.factory import ProviderName, get_default_model, get_default_provider, get_llm
from app.models.persistence import get_checkpointer


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


@lru_cache(maxsize=None)
def _compiled_chat_app(provider: str, model: str) -> Any:
    llm = get_llm(provider=ProviderName(provider), model=model)

    async def chatbot(state: ChatState) -> ChatState:
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chatbot", chatbot)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)

    return graph.compile(checkpointer=get_checkpointer())


def clear_cache() -> None:
    _compiled_chat_app.cache_clear()


async def get_chat_app_async(provider: str | None = None, model: str | None = None) -> Any:
    resolved_provider = provider or get_default_provider().value
    resolved_model = model or get_default_model(resolved_provider)
    return _compiled_chat_app(resolved_provider, resolved_model)


def _build_studio_graph(provider: str, model: str) -> Any:
    llm = get_llm(provider=ProviderName(provider), model=model)

    async def chatbot(state: ChatState) -> ChatState:
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}

    g = StateGraph(ChatState)
    g.add_node("chatbot", chatbot)
    g.add_edge(START, "chatbot")
    g.add_edge("chatbot", END)
    return g.compile()


graph = _build_studio_graph(get_default_provider().value, get_default_model())
