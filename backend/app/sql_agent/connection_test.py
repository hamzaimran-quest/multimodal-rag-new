"""Test user PostgreSQL connectivity."""

from __future__ import annotations

from sqlalchemy import create_engine, text

from app.config import settings


def test_postgres_connection(url: str) -> None:
    engine = create_engine(
        url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": int(settings.sql_agent_query_timeout_seconds)},
    )
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    finally:
        engine.dispose()
