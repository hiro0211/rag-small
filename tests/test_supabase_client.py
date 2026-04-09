from unittest.mock import patch, MagicMock
import pytest


def test_get_supabase_client_returns_client():
    with patch("lib.supabase_client.create_client") as mock_create:
        mock_create.return_value = MagicMock()
        from lib.supabase_client import get_supabase_client

        client = get_supabase_client()
        assert client is not None


def test_get_supabase_admin_returns_client():
    with patch("lib.supabase_client.create_client") as mock_create:
        mock_create.return_value = MagicMock()
        from lib.supabase_client import get_supabase_admin

        admin = get_supabase_admin()
        assert admin is not None


def test_get_supabase_client_uses_publishable_key():
    with patch("lib.supabase_client.create_client") as mock_create:
        mock_create.return_value = MagicMock()
        from lib.supabase_client import get_supabase_client

        get_supabase_client()
        mock_create.assert_called_with(
            "https://test.supabase.co", "test-publishable-key"
        )


def test_get_supabase_admin_uses_secret_key():
    with patch("lib.supabase_client.create_client") as mock_create:
        mock_create.return_value = MagicMock()
        from lib.supabase_client import get_supabase_admin

        get_supabase_admin()
        mock_create.assert_called_with(
            "https://test.supabase.co", "test-secret-key"
        )
