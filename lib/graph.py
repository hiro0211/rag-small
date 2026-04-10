"""LangGraph RAG pipeline with conversation history support."""

from typing import Annotated, Generator

from typing_extensions import TypedDict
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from lib.llm import create_llm, DEFAULT_MODEL
from lib.rag_chain import search_relevant_documents, RAG_SYSTEM_PROMPT


REWRITE_PROMPT = """あなたはユーザーの質問を検索用に書き換えるアシスタントです。
以下のルールで、元の質問の意図を保ったまま、検索に適した質問文に書き換えてください：

- 略語は正式名称に展開してください（例: ラグ → RAG (Retrieval-Augmented Generation)）
- ひらがなのタイポは正規化してください（例: べくとる → ベクトル）
- 曖昧・断片的な質問は完全な疑問文にしてください（例: ベクトルか → ベクトル検索とは何ですか）
- 元の質問が既に十分明確な場合は、そのまま返してください
- 出力は書き換え後の質問文のみ。説明や前置きは不要。

元の質問: {question}
"""


class RAGState(TypedDict):
    """State schema for the RAG LangGraph."""

    messages: Annotated[list[AnyMessage], add_messages]
    rewritten_query: str
    context: str
    sources: list
    model_id: str


def rewrite_query(state: RAGState) -> dict:
    """Rewrite the user query for better retrieval."""
    question = state["messages"][-1].content
    llm = create_llm(state.get("model_id") or DEFAULT_MODEL)
    prompt = REWRITE_PROMPT.format(question=question)
    response = llm.invoke([HumanMessage(content=prompt)])
    rewritten = response.content.strip() or question
    return {"rewritten_query": rewritten}


def retrieve(state: RAGState) -> dict:
    """Retrieve relevant documents using the rewritten query when available."""
    query = state.get("rewritten_query") or state["messages"][-1].content
    result = search_relevant_documents(query)
    return {"context": result["context"], "sources": result["sources"]}


def generate(state: RAGState) -> dict:
    """Generate response using LLM with conversation history and RAG context."""
    system_msg = SystemMessage(
        content=RAG_SYSTEM_PROMPT.format(context=state["context"])
    )

    trimmed = trim_messages(
        state["messages"],
        max_tokens=20,
        token_counter=len,
        strategy="last",
        start_on="human",
        include_system=False,
    )

    llm = create_llm(state.get("model_id") or DEFAULT_MODEL)
    response = llm.invoke([system_msg] + trimmed)
    return {"messages": [response]}


def build_rag_graph():
    """Build and compile the RAG LangGraph."""
    graph = StateGraph(RAGState)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def _build_messages(
    question: str, history: list[dict[str, str]]
) -> list[AnyMessage]:
    """Convert history dicts + question into LangChain messages."""
    messages: list[AnyMessage] = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))
    return messages


def stream_response(
    question: str, history: list[dict[str, str]], model_id: str = ""
) -> Generator[str, None, None]:
    """Stream LLM tokens from the RAG graph for Streamlit consumption."""
    messages = _build_messages(question, history)

    graph = build_rag_graph()
    for chunk in graph.stream(
        {
            "messages": messages,
            "rewritten_query": "",
            "context": "",
            "sources": [],
            "model_id": model_id,
        },
        stream_mode="messages",
        version="v2",
    ):
        if chunk["type"] == "messages":
            msg_chunk, metadata = chunk["data"]
            if msg_chunk.content and metadata.get("langgraph_node") == "generate":
                yield msg_chunk.content


def stream_response_with_sources(
    question: str, history: list[dict[str, str]], model_id: str = ""
) -> tuple[Generator[str, None, None], list]:
    """Stream LLM tokens and return sources from the RAG graph.

    Returns (token_generator, sources) tuple.
    Rewrites the query once to fetch sources immediately for the UI,
    token streaming is delegated to stream_response (which rewrites again
    inside the graph — see plan Part 3 note).
    """
    llm = create_llm(model_id or DEFAULT_MODEL)
    prompt = REWRITE_PROMPT.format(question=question)
    rewritten = llm.invoke([HumanMessage(content=prompt)]).content.strip() or question
    result = search_relevant_documents(rewritten)
    sources = result["sources"]
    return stream_response(question, history, model_id=model_id), sources
