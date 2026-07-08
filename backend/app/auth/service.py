"""User and refresh-token persistence operations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import security
from app.config import settings
from app.db.models import RefreshToken, User


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _aware(dt: datetime) -> datetime:
    """Treat naive timestamps (e.g. from SQLite) as UTC for safe comparison."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def get_user_by_email(db: Session, email: str) -> User | None:
    stmt = select(User).where(User.email == _normalize_email(email))
    return db.execute(stmt).scalar_one_or_none()


def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.get(User, user_id)


def create_user(db: Session, email: str, password: str) -> User:
    """Insert a new user. Caller is responsible for committing."""
    user = User(
        email=_normalize_email(email),
        password_hash=security.hash_password(password),
    )
    db.add(user)
    db.flush()  # populate user.id without committing
    return user


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    user = get_user_by_email(db, email)
    if user is None:
        return None
    if not security.verify_password(password, user.password_hash):
        return None
    return user


def issue_refresh_token(db: Session, user: User) -> str:
    """Create a refresh token row and return the raw token. Caller commits."""
    raw_token = security.generate_refresh_token()
    record = RefreshToken(
        user_id=user.id,
        token_hash=security.hash_refresh_token(raw_token),
        expires_at=datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_ttl_days),
    )
    db.add(record)
    db.flush()
    return raw_token


def _get_active_refresh(db: Session, raw_token: str) -> RefreshToken | None:
    token_hash = security.hash_refresh_token(raw_token)
    stmt = select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    record = db.execute(stmt).scalar_one_or_none()
    if record is None or record.revoked_at is not None:
        return None
    if _aware(record.expires_at) < datetime.now(timezone.utc):
        return None
    return record


def rotate_refresh_token(db: Session, raw_token: str) -> tuple[User, str] | None:
    """Validate + rotate: revoke the presented token and issue a fresh one."""
    record = _get_active_refresh(db, raw_token)
    if record is None:
        return None
    record.revoked_at = datetime.now(timezone.utc)
    user = db.get(User, record.user_id)
    if user is None:
        return None
    new_raw = issue_refresh_token(db, user)
    db.commit()
    return user, new_raw


def revoke_refresh_token(db: Session, raw_token: str) -> None:
    record = _get_active_refresh(db, raw_token)
    if record is not None:
        record.revoked_at = datetime.now(timezone.utc)
        db.commit()


def revoke_all_refresh_tokens(db: Session, user_id: int) -> int:
    """Revoke every active refresh token for a user. Returns count revoked."""
    now = datetime.now(timezone.utc)
    stmt = select(RefreshToken).where(
        RefreshToken.user_id == user_id,
        RefreshToken.revoked_at.is_(None),
    )
    records = list(db.scalars(stmt).all())
    for record in records:
        record.revoked_at = now
    return len(records)


def change_password(db: Session, user: User, current_password: str, new_password: str) -> bool:
    """Verify current password, set new hash, revoke all refresh tokens. Caller commits."""
    if not security.verify_password(current_password, user.password_hash):
        return False
    user.password_hash = security.hash_password(new_password)
    revoke_all_refresh_tokens(db, user.id)
    return True
