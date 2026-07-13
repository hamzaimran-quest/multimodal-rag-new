"""Fetch PostgreSQL schema digests for SQL routing and agents."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import create_engine, inspect

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SchemaSnapshot:
    digest: str
    tables: list[str]


def fetch_schema_snapshot(connection_url: str) -> SchemaSnapshot:
    """Introspect PostgreSQL and return a compact schema digest."""
    engine = create_engine(
        connection_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": int(settings.sql_agent_query_timeout_seconds)},
    )
    try:
        inspector = inspect(engine)
        schema_names = [name for name in inspector.get_schema_names() if name not in {"information_schema", "pg_catalog"}]
        if not schema_names:
            schema_names = ["public"]

        table_entries: list[tuple[str, str]] = []
        for schema_name in schema_names:
            for table_name in sorted(inspector.get_table_names(schema=schema_name)):
                qualified = f"{schema_name}.{table_name}" if schema_name != "public" else table_name
                columns = inspector.get_columns(table_name, schema=schema_name)
                col_parts = [f"{col['name']}:{_short_type(col.get('type'))}" for col in columns]
                table_entries.append((qualified, ", ".join(col_parts)))

        if len(table_entries) > settings.sql_schema_max_tables:
            table_entries = table_entries[: settings.sql_schema_max_tables]

        lines = [f"{name}({cols})" for name, cols in table_entries if cols]
        digest = "\n".join(lines)
        if len(digest) > settings.sql_schema_max_chars:
            digest = digest[: settings.sql_schema_max_chars].rstrip() + "\n..."

        tables = [name for name, _ in table_entries]
        logger.info("SQL schema fetched tables=%s digest_chars=%s", len(tables), len(digest))
        return SchemaSnapshot(digest=digest, tables=tables)
    finally:
        engine.dispose()


def _short_type(col_type: object | None) -> str:
    if col_type is None:
        return "unknown"
    name = getattr(col_type, "__visit_name__", None) or str(col_type)
    return str(name).lower()
