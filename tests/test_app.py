"""Tests for the Streamlit chat app logic."""

from unittest.mock import patch, MagicMock
import pytest


class TestGenerateResponse:
    def test_calls_search_and_openai(self):
        with (
            patch("lib.chat.search_relevant_documents") as mock_search,
            patch("lib.chat.OpenAI") as MockOpenAI,
        ):
            mock_search.return_value = {
                "context": "RAGはRetrieval Augmented Generationの略です。",
                "sources": [],
            }
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client

            # Mock streaming response
            mock_chunk = MagicMock()
            mock_chunk.choices = [MagicMock()]
            mock_chunk.choices[0].delta.content = "回答テスト"
            mock_client.chat.completions.create.return_value = [mock_chunk]

            from lib.chat import generate_response

            chunks = list(generate_response("RAGとは？"))
            assert len(chunks) > 0
            mock_search.assert_called_once_with("RAGとは？")

    def test_uses_gpt4o_mini(self):
        with (
            patch("lib.chat.search_relevant_documents") as mock_search,
            patch("lib.chat.OpenAI") as MockOpenAI,
        ):
            mock_search.return_value = {"context": "", "sources": []}
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = []

            from lib.chat import generate_response

            list(generate_response("test"))

            call_kwargs = mock_client.chat.completions.create.call_args
            assert call_kwargs.kwargs["model"] == "gpt-4o-mini"
            assert call_kwargs.kwargs["temperature"] == 0
            assert call_kwargs.kwargs["stream"] is True

    def test_returns_no_context_message_when_empty(self):
        with (
            patch("lib.chat.search_relevant_documents") as mock_search,
            patch("lib.chat.OpenAI") as MockOpenAI,
        ):
            mock_search.return_value = {"context": "", "sources": []}
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client

            mock_chunk = MagicMock()
            mock_chunk.choices = [MagicMock()]
            mock_chunk.choices[0].delta.content = "ナレッジベースに含まれていません"
            mock_client.chat.completions.create.return_value = [mock_chunk]

            from lib.chat import generate_response

            chunks = list(generate_response("今日の天気は？"))
            assert len(chunks) > 0
