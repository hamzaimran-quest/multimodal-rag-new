"""Authentication API: signup, login, token refresh, logout, current user."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth import security, service
from app.auth.dependencies import get_current_user
from app.auth.schemas import ChangePasswordRequest, LoginRequest, SignupRequest, TokenResponse, UserResponse
from app.config import settings
from app.db.models import User
from app.db.session import get_db

router = APIRouter(prefix="/auth", tags=["auth"])

# Refresh cookie is scoped to /auth so it is only sent on refresh/logout, never on
# the access-token-authenticated API surface.
_COOKIE_PATH = "/auth"


def _set_refresh_cookie(response: Response, raw_token: str) -> None:
    response.set_cookie(
        key=settings.refresh_cookie_name,
        value=raw_token,
        max_age=settings.refresh_token_ttl_seconds,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        path=_COOKIE_PATH,
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(
        key=settings.refresh_cookie_name,
        path=_COOKIE_PATH,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
    )


def _token_response(user: User, access_token: str) -> TokenResponse:
    return TokenResponse(
        access_token=access_token,
        user=UserResponse(id=user.id, email=user.email),
    )


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def signup(body: SignupRequest, response: Response, db: Session = Depends(get_db)) -> TokenResponse:
    if service.get_user_by_email(db, body.email) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    user = service.create_user(db, body.email, body.password)
    raw_refresh = service.issue_refresh_token(db, user)
    try:
        db.commit()
    except IntegrityError as exc:  # unique-constraint backstop against a race
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Email already registered"
        ) from exc

    _set_refresh_cookie(response, raw_refresh)
    return _token_response(user, security.create_access_token(user.id))


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, response: Response, db: Session = Depends(get_db)) -> TokenResponse:
    user = service.authenticate_user(db, body.email, body.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
        )

    raw_refresh = service.issue_refresh_token(db, user)
    db.commit()

    _set_refresh_cookie(response, raw_refresh)
    return _token_response(user, security.create_access_token(user.id))


@router.post("/refresh", response_model=TokenResponse)
def refresh(request: Request, response: Response, db: Session = Depends(get_db)) -> TokenResponse:
    raw_refresh = request.cookies.get(settings.refresh_cookie_name)
    if not raw_refresh:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing refresh token"
        )

    rotated = service.rotate_refresh_token(db, raw_refresh)
    if rotated is None:
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token"
        )

    user, new_refresh = rotated
    _set_refresh_cookie(response, new_refresh)
    return _token_response(user, security.create_access_token(user.id))


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)) -> dict[str, str]:
    raw_refresh = request.cookies.get(settings.refresh_cookie_name)
    if raw_refresh:
        service.revoke_refresh_token(db, raw_refresh)
    _clear_refresh_cookie(response)
    return {"status": "logged out"}


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse(id=current_user.id, email=current_user.email)


@router.post("/change-password")
def change_password(
    body: ChangePasswordRequest,
    response: Response,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    if body.current_password == body.new_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must differ from current password",
        )

    updated = service.change_password(
        db,
        current_user,
        body.current_password,
        body.new_password,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect",
        )

    db.commit()
    _clear_refresh_cookie(response)
    return {"status": "password_changed"}
