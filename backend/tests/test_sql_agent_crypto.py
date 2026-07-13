"""Tests for SQL connection URL validation and encryption."""

from __future__ import annotations

import pytest

from app.sql_agent.crypto import decrypt_connection_url, encrypt_connection_url
from app.sql_agent.url import validate_postgres_url


def test_validate_postgres_url_normalizes_scheme() -> None:
    url = validate_postgres_url("postgresql://user:pass@localhost:5432/analytics")
    assert url.startswith("postgresql+psycopg2://")
    assert url.endswith("/analytics")


def test_validate_postgres_url_rejects_mysql() -> None:
    with pytest.raises(ValueError, match="postgresql_only"):
        validate_postgres_url("mysql://user:pass@localhost/db")


def test_encrypt_decrypt_roundtrip() -> None:
    original = "postgresql+psycopg2://user:secret@localhost:5432/demo"
    token = encrypt_connection_url(original)
    assert token != original
    assert decrypt_connection_url(token) == original
