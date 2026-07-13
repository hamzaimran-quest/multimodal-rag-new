"""Per-connection schema cache (memory + DB persistence)."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import UserSqlConnection
from app.sql_agent.schema_fetch import SchemaSnapshot, fetch_schema_snapshot

logger = logging.getLogger(__name__)

_memory_lock = Lock()
_memory_cache: dict[tuple[int, int, str], SchemaSnapshot] = {}


@dataclass(frozen=True)
class CachedSchema:
    digest: str
    tables: list[str]
    fingerprint: str
    cached_at: datetime
    source: str


def credentials_fingerprint(connection_url: str) -> str:
    return hashlib.sha256(connection_url.encode("utf-8")).hexdigest()


def forget_memory(*, user_id: int, connection_id: int) -> None:
    with _memory_lock:
        keys = [key for key in _memory_cache if key[0] == user_id and key[1] == connection_id]
        for key in keys:
            _memory_cache.pop(key, None)


def invalidate_persisted_cache(row: UserSqlConnection) -> None:
    row.schema_cache = None
    row.schema_cache_fingerprint = None
    row.schema_cached_at = None
    forget_memory(user_id=row.user_id, connection_id=row.id)


def _snapshot_from_row(row: UserSqlConnection, fingerprint: str) -> CachedSchema | None:
    payload = row.schema_cache
    if not isinstance(payload, dict):
        return None
    if row.schema_cache_fingerprint != fingerprint:
        return None
    digest = str(payload.get("digest") or "").strip()
    if not digest:
        return None
    tables_raw = payload.get("tables") or []
    tables = [str(name) for name in tables_raw if name]
    cached_at = row.schema_cached_at or datetime.now(UTC)
    return CachedSchema(
        digest=digest,
        tables=tables,
        fingerprint=fingerprint,
        cached_at=cached_at,
        source="db",
    )


def _persist_snapshot(db: Session, row: UserSqlConnection, snapshot: SchemaSnapshot, fingerprint: str) -> CachedSchema:
    now = datetime.now(UTC)
    row.schema_cache = {"digest": snapshot.digest, "tables": snapshot.tables}
    row.schema_cache_fingerprint = fingerprint
    row.schema_cached_at = now
    db.flush()
    return CachedSchema(
        digest=snapshot.digest,
        tables=list(snapshot.tables),
        fingerprint=fingerprint,
        cached_at=now,
        source="fresh",
    )


def _memory_get(user_id: int, connection_id: int, fingerprint: str) -> SchemaSnapshot | None:
    with _memory_lock:
        snapshot = _memory_cache.get((user_id, connection_id, fingerprint))
    return snapshot


def _memory_set(user_id: int, connection_id: int, fingerprint: str, snapshot: SchemaSnapshot) -> None:
    with _memory_lock:
        _memory_cache[(user_id, connection_id, fingerprint)] = snapshot


def _is_stale(cached_at: datetime) -> bool:
    ttl = int(settings.sql_connection_cache_ttl_seconds)
    if ttl <= 0:
        return False
    age = (datetime.now(UTC) - cached_at).total_seconds()
    return age > ttl


def get_or_load_schema(
    db: Session,
    row: UserSqlConnection,
    connection_url: str,
    *,
    force_refresh: bool = False,
) -> CachedSchema:
    """Return cached schema for a connection, fetching from PostgreSQL on miss."""
    fingerprint = credentials_fingerprint(connection_url)

    if not force_refresh:
        memory_snapshot = _memory_get(row.user_id, row.id, fingerprint)
        if memory_snapshot is not None:
            return CachedSchema(
                digest=memory_snapshot.digest,
                tables=list(memory_snapshot.tables),
                fingerprint=fingerprint,
                cached_at=datetime.now(UTC),
                source="memory",
            )

        persisted = _snapshot_from_row(row, fingerprint)
        if persisted is not None and not _is_stale(persisted.cached_at):
            _memory_set(
                row.user_id,
                row.id,
                fingerprint,
                SchemaSnapshot(digest=persisted.digest, tables=persisted.tables),
            )
            return persisted

    snapshot = fetch_schema_snapshot(connection_url)
    _memory_set(row.user_id, row.id, fingerprint, snapshot)
    cached = _persist_snapshot(db, row, snapshot, fingerprint)
    logger.info(
        "SQL schema_cache refreshed user_id=%s connection_id=%s tables=%s",
        row.user_id,
        row.id,
        len(snapshot.tables),
    )
    return cached


def warm_schema_cache(db: Session, row: UserSqlConnection, connection_url: str) -> CachedSchema:
    return get_or_load_schema(db, row, connection_url, force_refresh=True)
