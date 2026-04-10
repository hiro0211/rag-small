"""Tests for lib/llm.py - LLM factory for switching between providers."""

import pytest
from unittest.mock import patch, MagicMock


class TestCreateLLM:
    """Tests for create_llm() factory."""

    @patch("lib.llm.ChatOpenAI")
    def test_creates_openai_for_gpt4o_mini(self, mock_openai_cls):
        from lib.llm import create_llm

        mock_instance = MagicMock()
        mock_openai_cls.return_value = mock_instance

        result = create_llm("gpt-4o-mini")

        mock_openai_cls.assert_called_once_with(
            model="gpt-4o-mini", temperature=0, streaming=True
        )
        assert result is mock_instance

    @patch("lib.llm.ChatGoogleGenerativeAI")
    def test_creates_gemini_for_gemini_flash(self, mock_gemini_cls):
        from lib.llm import create_llm

        mock_instance = MagicMock()
        mock_gemini_cls.return_value = mock_instance

        result = create_llm("gemini-2.5-flash")

        mock_gemini_cls.assert_called_once_with(
            model="gemini-2.5-flash", temperature=0
        )
        assert result is mock_instance

    def test_raises_for_unknown_model(self):
        from lib.llm import create_llm

        with pytest.raises(ValueError, match="Unknown model"):
            create_llm("unknown-model-id")

    @patch("lib.llm.ChatOpenAI")
    def test_empty_string_uses_default(self, mock_openai_cls):
        from lib.llm import create_llm

        mock_openai_cls.return_value = MagicMock()

        create_llm("")

        mock_openai_cls.assert_called_once_with(
            model="gpt-4o-mini", temperature=0, streaming=True
        )


class TestGetAvailableModels:
    """Tests for get_available_models() helper."""

    def test_returns_model_dict(self):
        from lib.llm import get_available_models

        models = get_available_models()

        assert isinstance(models, dict)
        assert "GPT-4o-mini" in models
        assert "Gemini 2.5 Flash" in models
        assert models["GPT-4o-mini"] == "gpt-4o-mini"
        assert models["Gemini 2.5 Flash"] == "gemini-2.5-flash"


class TestDefaultModel:
    """Tests for DEFAULT_MODEL constant."""

    def test_default_model_is_gpt4o_mini(self):
        from lib.llm import DEFAULT_MODEL

        assert DEFAULT_MODEL == "gpt-4o-mini"
