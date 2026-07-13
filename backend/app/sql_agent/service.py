"""CRUD for per-user SQL connections."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.models import UserSqlConnection
from app.sql_agent import schema_cache
from app.sql_agent.connection_test import test_postgres_connection
from app.sql_agent.crypto import decrypt_connection_url, encrypt_connection_url
from app.sql_agent.models import ActiveSqlSnapshot
from app.sql_agent.url import validate_postgres_url


def list_connections(db: Session, user_id: int) -> list[UserSqlConnection]:
    stmt = (
        select(UserSqlConnection)
        .where(UserSqlConnection.user_id == user_id)
        .order_by(UserSqlConnection.created_at.asc())
    )
    return list(db.scalars(stmt).all())


def get_connection_for_user(db: Session, connection_id: int, user_id: int) -> UserSqlConnection | None:
    stmt = select(UserSqlConnection).where(
        UserSqlConnection.id == connection_id,
        UserSqlConnection.user_id == user_id,
    )
    return db.scalars(stmt).first()


def get_active_connection(db: Session, user_id: int) -> UserSqlConnection | None:
    stmt = select(UserSqlConnection).where(
        UserSqlConnection.user_id == user_id,
        UserSqlConnection.is_active.is_(True),
    )
    return db.scalars(stmt).first()


def _deactivate_all(db: Session, user_id: int) -> None:
    db.execute(
        update(UserSqlConnection)
        .where(UserSqlConnection.user_id == user_id, UserSqlConnection.is_active.is_(True))
        .values(is_active=False)
    )


def add_connection(
    db: Session,
    *,
    user_id: int,
    connection_url: str,
    display_name: str,
    description: str,
    activate: bool = False,
) -> UserSqlConnection:
    normalized = validate_postgres_url(connection_url)
    test_postgres_connection(normalized)

    existing = list_connections(db, user_id)
    should_activate = activate or len(existing) == 0
    if should_activate:
        _deactivate_all(db, user_id)

    now = datetime.now(UTC)
    row = UserSqlConnection(
        user_id=user_id,
        display_name=display_name.strip(),
        description=description.strip(),
        connection_url_encrypted=encrypt_connection_url(normalized),
        dialect="postgresql",
        is_active=should_activate,
        last_tested_at=now,
        last_error=None,
    )
    db.add(row)
    db.flush()
    _refresh_schema_cache(db, row)
    return row


def activate_connection(db: Session, *, user_id: int, connection_id: int) -> UserSqlConnection:
    row = get_connection_for_user(db, connection_id, user_id)
    if row is None:
        raise LookupError("connection_not_found")
    _deactivate_all(db, user_id)
    row.is_active = True
    row.last_error = None
    db.flush()
    _refresh_schema_cache(db, row)
    return row


def deactivate_all(db: Session, user_id: int) -> int:
    result = db.execute(
        update(UserSqlConnection)
        .where(UserSqlConnection.user_id == user_id, UserSqlConnection.is_active.is_(True))
        .values(is_active=False)
    )
    return int(result.rowcount or 0)


def delete_connection(db: Session, *, user_id: int, connection_id: int) -> bool:
    row = get_connection_for_user(db, connection_id, user_id)
    if row is None:
        return False
    schema_cache.forget_memory(user_id=row.user_id, connection_id=row.id)
    db.delete(row)
    db.flush()
    return True


def update_metadata(
    db: Session,
    *,
    user_id: int,
    connection_id: int,
    display_name: str | None = None,
    description: str | None = None,
) -> UserSqlConnection:
    row = get_connection_for_user(db, connection_id, user_id)
    if row is None:
        raise LookupError("connection_not_found")
    if display_name is not None:
        row.display_name = display_name.strip()
    if description is not None:
        row.description = description.strip()
    db.flush()
    return row


def update_credentials(
    db: Session,
    *,
    user_id: int,
    connection_id: int,
    connection_url: str,
) -> UserSqlConnection:
    row = get_connection_for_user(db, connection_id, user_id)
    if row is None:
        raise LookupError("connection_not_found")
    normalized = validate_postgres_url(connection_url)
    test_postgres_connection(normalized)
    schema_cache.invalidate_persisted_cache(row)
    row.connection_url_encrypted = encrypt_connection_url(normalized)
    row.last_tested_at = datetime.now(UTC)
    row.last_error = None
    db.flush()
    _refresh_schema_cache(db, row)
    return row


def test_saved_connection(db: Session, *, user_id: int, connection_id: int) -> UserSqlConnection:
    row = get_connection_for_user(db, connection_id, user_id)
    if row is None:
        raise LookupError("connection_not_found")
    url = decrypt_connection_url(row.connection_url_encrypted)
    try:
        test_postgres_connection(url)
        row.last_tested_at = datetime.now(UTC)
        row.last_error = None
    except Exception as exc:
        row.last_error = str(exc)[:500]
        db.flush()
        raise
    db.flush()
    return row


def connection_url_for_row(row: UserSqlConnection) -> str:
    return decrypt_connection_url(row.connection_url_encrypted)


def snapshot_connection(row: UserSqlConnection) -> ActiveSqlSnapshot:
    """Copy fields needed after the SQLAlchemy session is closed."""
    return ActiveSqlSnapshot(
        connection_id=row.id,
        display_name=row.display_name,
        description=row.description or "",
    )


def _refresh_schema_cache(db: Session, row: UserSqlConnection) -> None:
    schema_cache.warm_schema_cache(db, row, connection_url_for_row(row))
