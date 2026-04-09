"""Chat logic: search documents, build prompt, stream response."""

from typing import Generator
from openai import OpenAI
from lib.rag_chain import search_relevant_documents, build_rag_prompt


def generate_response(question: str) -> Generator[str, None, None]:
    """Search relevant docs and stream GPT-4o-mini response."""
    result = search_relevant_documents(question)
    messages = build_rag_prompt(question=question, context=result["context"])

    client = OpenAI()
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
