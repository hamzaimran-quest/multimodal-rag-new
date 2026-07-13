"""Tests for schema-first SQL routing gate."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.models import User, UserSqlConnection
from app.sql_agent.routing import plan_sql_route
from app.sql_agent.schema_cache import CachedSchema
from app.sql_agent.scope_classifier import ScopeClassification
from app.sql_agent.service import snapshot_connection


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
    user = User(email="route@example.com", password_hash="x")
    session.add(user)
    session.flush()
    row = UserSqlConnection(
        user_id=user.id,
        display_name="DVD",
        description="dvd rental",
        connection_url_encrypted="enc",
        dialect="postgresql",
        is_active=True,
    )
    session.add(row)
    session.commit()
    yield session, row
    session.close()
    engine.dispose()


@pytest.mark.asyncio
@patch("app.sql_agent.routing.classify_query_scope", new_callable=AsyncMock)
@patch("app.sql_agent.routing.schema_cache.get_or_load_schema")
async def test_plan_sql_route_film_query(mock_cache, mock_classify, db_session) -> None:
    session, row = db_session
    mock_cache.return_value = CachedSchema(
        digest="film(id, title)",
        tables=["film", "actor"],
        fingerprint="abc",
        cached_at=datetime.now(UTC),
        source="memory",
    )
    mock_classify.return_value = ScopeClassification(
        decision="sql",
        confidence=0.9,
        matched_tables=["film"],
        reason="film lookup",
    )

    plan = await plan_sql_route(
        db=session,
        query="tell me about the film Ace Goldfinger",
        active_sql=row,
        connection_url="postgresql://u:p@host/db",
    )
    assert plan.mode == "sql"
    assert "film" in plan.matched_tables


@pytest.mark.asyncio
async def test_plan_sql_route_without_active_connection() -> None:
    plan = await plan_sql_route(
        db=MagicMock(),
        query="hello",
        active_sql=None,
        connection_url=None,
    )
    assert plan.mode == "rag"


def test_snapshot_connection_survives_session_close(db_session) -> None:
    session, row = db_session
    snap = snapshot_connection(row)
    session.close()
    assert snap.connection_id == row.id
    assert snap.display_name == "DVD"
    assert snap.description == "dvd rental"
