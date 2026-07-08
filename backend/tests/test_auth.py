"""Tests for Phase 1 authentication: signup, login, refresh, logout, me.

These tests use an in-memory SQLite database via a dependency override, so they
do not require PostgreSQL to be running.
"""

from __future__ import annotations

import jwt
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth import security
from app.config import settings
from app.db import models  # noqa: F401  (registers tables on Base.metadata)
from app.db.base import Base
from app.db.session import get_db
from app.main import app

REFRESH_COOKIE = settings.refresh_cookie_name


@pytest.fixture
async def auth_client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.pop(get_db, None)
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


async def _signup(client: AsyncClient, email="a@example.com", password="supersecret1"):
    return await client.post("/auth/signup", json={"email": email, "password": password})


# --- security unit tests (no DB) ---------------------------------------------


def test_password_hash_roundtrip():
    hashed = security.hash_password("supersecret1")
    assert hashed != "supersecret1"
    assert security.verify_password("supersecret1", hashed) is True
    assert security.verify_password("wrong", hashed) is False


def test_access_token_encodes_subject_and_type():
    token = security.create_access_token(42)
    payload = security.decode_access_token(token)
    assert payload["sub"] == "42"
    assert payload["type"] == "access"


def test_expired_access_token_rejected():
    token = jwt.encode(
        {"sub": "1", "type": "access", "exp": 0},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(jwt.ExpiredSignatureError):
        security.decode_access_token(token)


def test_refresh_token_hash_is_deterministic_and_hidden():
    raw = security.generate_refresh_token()
    assert security.hash_refresh_token(raw) == security.hash_refresh_token(raw)
    assert security.hash_refresh_token(raw) != raw


# --- signup ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_signup_returns_token_and_sets_cookie(auth_client):
    response = await _signup(auth_client)
    assert response.status_code == 201
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["user"]["email"] == "a@example.com"
    assert REFRESH_COOKIE in response.cookies


@pytest.mark.asyncio
async def test_signup_duplicate_email_conflicts(auth_client):
    await _signup(auth_client)
    dup = await _signup(auth_client)
    assert dup.status_code == 409


@pytest.mark.asyncio
async def test_signup_normalizes_email_case(auth_client):
    await _signup(auth_client, email="Mixed@Example.com")
    dup = await _signup(auth_client, email="mixed@example.com")
    assert dup.status_code == 409


@pytest.mark.asyncio
async def test_signup_rejects_short_password(auth_client):
    response = await auth_client.post(
        "/auth/signup", json={"email": "b@example.com", "password": "short"}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_signup_rejects_invalid_email(auth_client):
    response = await auth_client.post(
        "/auth/signup", json={"email": "not-an-email", "password": "supersecret1"}
    )
    assert response.status_code == 422


# --- login -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_login_success(auth_client):
    await _signup(auth_client)
    response = await auth_client.post(
        "/auth/login", json={"email": "a@example.com", "password": "supersecret1"}
    )
    assert response.status_code == 200
    assert response.json()["access_token"]
    assert REFRESH_COOKIE in response.cookies


@pytest.mark.asyncio
async def test_login_wrong_password_rejected(auth_client):
    await _signup(auth_client)
    response = await auth_client.post(
        "/auth/login", json={"email": "a@example.com", "password": "wrongpassword"}
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_login_unknown_email_rejected(auth_client):
    response = await auth_client.post(
        "/auth/login", json={"email": "nobody@example.com", "password": "supersecret1"}
    )
    assert response.status_code == 401


# --- me ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_me_returns_current_user(auth_client):
    token = (await _signup(auth_client)).json()["access_token"]
    response = await auth_client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["email"] == "a@example.com"


@pytest.mark.asyncio
async def test_me_without_token_unauthorized(auth_client):
    response = await auth_client.get("/auth/me")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_me_with_garbage_token_unauthorized(auth_client):
    response = await auth_client.get(
        "/auth/me", headers={"Authorization": "Bearer not.a.jwt"}
    )
    assert response.status_code == 401


# --- refresh + logout --------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_issues_new_access_token(auth_client):
    await _signup(auth_client)
    response = await auth_client.post("/auth/refresh")
    assert response.status_code == 200
    assert response.json()["access_token"]
    assert REFRESH_COOKIE in response.cookies


@pytest.mark.asyncio
async def test_refresh_without_cookie_unauthorized(auth_client):
    response = await auth_client.post("/auth/refresh")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_rotated_refresh_token_cannot_be_reused(auth_client):
    await _signup(auth_client)
    old_value = auth_client.cookies.get(REFRESH_COOKIE)

    rotated = await auth_client.post("/auth/refresh")
    assert rotated.status_code == 200

    # Replay the pre-rotation token: it was revoked during rotation.
    auth_client.cookies.clear()
    auth_client.cookies.set(REFRESH_COOKIE, old_value, domain="test", path="/auth")
    replay = await auth_client.post("/auth/refresh")
    assert replay.status_code == 401


@pytest.mark.asyncio
async def test_logout_revokes_refresh_token(auth_client):
    await _signup(auth_client)
    logged_out = await auth_client.post("/auth/logout")
    assert logged_out.status_code == 200

    after = await auth_client.post("/auth/refresh")
    assert after.status_code == 401


# --- change password ---------------------------------------------------------


@pytest.mark.asyncio
async def test_change_password_revokes_refresh_sessions(auth_client):
    signup = await _signup(auth_client)
    token = signup.json()["access_token"]
    old_refresh = auth_client.cookies.get(REFRESH_COOKIE)

    changed = await auth_client.post(
        "/auth/change-password",
        json={"current_password": "supersecret1", "new_password": "newsecret12"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert changed.status_code == 200
    assert changed.json()["status"] == "password_changed"

    auth_client.cookies.clear()
    auth_client.cookies.set(REFRESH_COOKIE, old_refresh, domain="test", path="/auth")
    replay = await auth_client.post("/auth/refresh")
    assert replay.status_code == 401

    old_login = await auth_client.post(
        "/auth/login", json={"email": "a@example.com", "password": "supersecret1"}
    )
    assert old_login.status_code == 401

    new_login = await auth_client.post(
        "/auth/login", json={"email": "a@example.com", "password": "newsecret12"}
    )
    assert new_login.status_code == 200


@pytest.mark.asyncio
async def test_change_password_rejects_wrong_current(auth_client):
    token = (await _signup(auth_client)).json()["access_token"]
    response = await auth_client.post(
        "/auth/change-password",
        json={"current_password": "wrongpassword", "new_password": "newsecret12"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_change_password_requires_auth(auth_client):
    response = await auth_client.post(
        "/auth/change-password",
        json={"current_password": "supersecret1", "new_password": "newsecret12"},
    )
    assert response.status_code == 401

