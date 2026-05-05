from functools import lru_cache
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from model_factory import ProviderName, get_chat_model


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def _build_chat_app(
    provider: ProviderName = "openrouter",
    model_name: str | None = None,
):
    llm = get_chat_model(provider=provider, model_name=model_name)

    def chatbot(state: ChatState) -> ChatState:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    checkpoint = InMemorySaver()

    graph = StateGraph(ChatState)
    graph.add_node("chatbot", chatbot)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)

    return graph.compile(checkpointer=checkpoint)


@lru_cache(maxsize=16)
def get_chat_app(
    provider: ProviderName = "openrouter",
    model_name: str | None = None,
):
    return _build_chat_app(provider=provider, model_name=model_name)


class LazyChatApp:
    def invoke(self, *args, **kwargs):
        return get_chat_app().invoke(*args, **kwargs)


app = LazyChatApp()
