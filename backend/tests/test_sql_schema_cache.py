"""Tests for SQL schema cache."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.models import User, UserSqlConnection
from app.sql_agent import schema_cache
from app.sql_agent.schema_fetch import SchemaSnapshot


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    user = User(email="cache@example.com", password_hash="x")
    session.add(user)
    session.flush()
    row = UserSqlConnection(
        user_id=user.id,
        display_name="DVD",
        description="rental schema",
        connection_url_encrypted="enc",
        dialect="postgresql",
        is_active=True,
    )
    session.add(row)
    session.commit()
    yield session, row
    session.close()
    engine.dispose()


@patch("app.sql_agent.schema_cache.fetch_schema_snapshot")
def test_schema_cache_persists_and_reuses(mock_fetch, db_session) -> None:
    session, row = db_session
    mock_fetch.return_value = SchemaSnapshot(digest="film(id, title)", tables=["film"])

    first = schema_cache.get_or_load_schema(session, row, "postgresql://u:p@host/db", force_refresh=True)
    session.commit()
    assert first.source == "fresh"
    assert first.tables == ["film"]

    mock_fetch.reset_mock()
    second = schema_cache.get_or_load_schema(session, row, "postgresql://u:p@host/db")
    assert second.source == "memory"
    mock_fetch.assert_not_called()


@patch("app.sql_agent.schema_cache.fetch_schema_snapshot")
def test_schema_cache_invalidates_on_fingerprint_change(mock_fetch, db_session) -> None:
    session, row = db_session
    mock_fetch.return_value = SchemaSnapshot(digest="film(id)", tables=["film"])
    schema_cache.get_or_load_schema(session, row, "postgresql://u:p@host/db1", force_refresh=True)
    session.commit()

    mock_fetch.return_value = SchemaSnapshot(digest="actor(id)", tables=["actor"])
    schema_cache.invalidate_persisted_cache(row)
    refreshed = schema_cache.get_or_load_schema(session, row, "postgresql://u:p@host/db2", force_refresh=True)
    assert refreshed.tables == ["actor"]
