"""FastAPI dependencies for extracting the authenticated user from a request."""

from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.auth import security, service
from app.db.models import User
from app.db.session import get_db

_bearer_scheme = HTTPBearer(auto_error=False)

_UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None or not credentials.credentials:
        raise _UNAUTHENTICATED

    try:
        payload = security.decode_access_token(credentials.credentials)
    except jwt.PyJWTError as exc:
        raise _UNAUTHENTICATED from exc

    if payload.get("type") != "access":
        raise _UNAUTHENTICATED

    subject = payload.get("sub")
    if subject is None:
        raise _UNAUTHENTICATED

    try:
        user_id = int(subject)
    except (TypeError, ValueError) as exc:
        raise _UNAUTHENTICATED from exc

    user = service.get_user_by_id(db, user_id)
    if user is None:
        raise _UNAUTHENTICATED
    return user
