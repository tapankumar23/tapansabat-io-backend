from functools import lru_cache
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from chat_persistence import get_checkpointer, initialize
from model_factory import DEFAULT_PROVIDER, ProviderName, get_chat_model


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


@lru_cache(maxsize=16)
def _compiled_chat_app(
    provider: ProviderName = DEFAULT_PROVIDER,
    model_name: str | None = None,
) -> Any:
    llm = get_chat_model(provider=provider, model_name=model_name)

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


async def get_chat_app_async(
    provider: ProviderName = DEFAULT_PROVIDER,
    model_name: str | None = None,
) -> Any:
    await initialize()
    return _compiled_chat_app(provider=provider, model_name=model_name)
