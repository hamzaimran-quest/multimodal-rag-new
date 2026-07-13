"""Database engine, session factory, and FastAPI session dependency."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.db.base import Base

engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db() -> Generator[Session, None, None]:
    """Yield a request-scoped database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables for all registered models. Safe to call repeatedly."""
    from app.db import models  # noqa: F401  (import registers tables on Base.metadata)

    Base.metadata.create_all(bind=engine)
    _ensure_optional_columns()


def _ensure_optional_columns() -> None:
    """Lightweight schema patches for dev/docker without Alembic."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if "chat_messages" in inspector.get_table_names():
        columns = {col["name"] for col in inspector.get_columns("chat_messages")}
        if "sql_meta" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE chat_messages ADD COLUMN sql_meta JSON"))

    if "user_sql_connections" in inspector.get_table_names():
        columns = {col["name"] for col in inspector.get_columns("user_sql_connections")}
        patches: list[str] = []
        if "schema_cache" not in columns:
            patches.append("ALTER TABLE user_sql_connections ADD COLUMN schema_cache JSON")
        if "schema_cache_fingerprint" not in columns:
            patches.append("ALTER TABLE user_sql_connections ADD COLUMN schema_cache_fingerprint VARCHAR(64)")
        if "schema_cached_at" not in columns:
            patches.append("ALTER TABLE user_sql_connections ADD COLUMN schema_cached_at TIMESTAMP WITH TIME ZONE")
        if patches:
            with engine.begin() as conn:
                for statement in patches:
                    conn.execute(text(statement))
