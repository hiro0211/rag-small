"""Chat logic: delegates to LangGraph RAG pipeline."""

from typing import Generator

from lib.graph import stream_response, stream_response_with_sources


def generate_response(
    question: str,
    history: list[dict[str, str]] | None = None,
    model_id: str = "",
) -> Generator[str, None, None]:
    """Stream LLM response using LangGraph RAG pipeline."""
    yield from stream_response(question, history or [], model_id=model_id)


def generate_response_with_sources(
    question: str,
    history: list[dict[str, str]] | None = None,
    model_id: str = "",
) -> tuple[Generator[str, None, None], list]:
    """Stream response and return sources for UI display."""
    return stream_response_with_sources(question, history or [], model_id=model_id)
