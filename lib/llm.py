"""LLM factory: switch between OpenAI GPT-4o-mini and Google Gemini 2.5 Flash."""

from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


DEFAULT_MODEL = "gpt-4o-mini"

AVAILABLE_MODELS: dict[str, str] = {
    "GPT-4o-mini": "gpt-4o-mini",
    "Gemini 2.5 Flash": "gemini-2.5-flash",
}


@lru_cache(maxsize=4)
def create_llm(model_id: str) -> BaseChatModel:
    """Return a chat model instance for the given model id.

    Falls back to DEFAULT_MODEL when model_id is an empty string.
    Raises ValueError on unknown ids.
    """
    resolved = model_id or DEFAULT_MODEL

    if resolved == "gpt-4o-mini":
        return ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

    if resolved == "gemini-2.5-flash":
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    raise ValueError(f"Unknown model id: {resolved}")


def get_available_models() -> dict[str, str]:
    """Return mapping of display names to model ids."""
    return dict(AVAILABLE_MODELS)
