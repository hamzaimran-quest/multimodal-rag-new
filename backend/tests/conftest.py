"""Shared pytest fixtures."""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from opensearchpy import OpenSearch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.opensearch.bootstrap import bootstrap_opensearch, wait_for_opensearch
from app.opensearch.client import get_opensearch_client

TEST_PASSWORD = "supersecret1"


def opensearch_available() -> bool:
    try:
        client = get_opensearch_client()
        client.cluster.health()
        client.close()
        return True
    except Exception:
        return False


requires_opensearch = pytest.mark.skipif(
    not opensearch_available(),
    reason="OpenSearch is not reachable at configured host/port",
)


@pytest.fixture(autouse=True)
def relax_jwt_requirement_for_tests(monkeypatch):
    monkeypatch.setattr(settings, "require_secure_jwt_secret", False)


@pytest.fixture
def auth_db_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def auth_db_session_factory(auth_db_engine):
    return sessionmaker(bind=auth_db_engine, autoflush=False, autocommit=False)


@pytest.fixture
def auth_db_override(auth_db_session_factory):
    def override_get_db():
        db = auth_db_session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def opensearch_client() -> OpenSearch:
    client = wait_for_opensearch()
    bootstrap_opensearch(client)
    yield client
    client.delete_by_query(
        index=settings.chunks_index,
        body={"query": {"term": {"extra_metadata.extraction_method": "test"}}},
        refresh=True,
        ignore=[404],
    )
    client.close()


@pytest.fixture
async def api_client():
    """Lightweight client for endpoints that do not require OpenSearch."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
async def api_client_with_opensearch(opensearch_client: OpenSearch, auth_db_override):
    """Authenticated client with OpenSearch + in-memory auth DB."""
    app.state.opensearch = opensearch_client
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        email = f"user-{uuid.uuid4().hex[:8]}@example.com"
        signup = await client.post(
            "/auth/signup",
            json={"email": email, "password": TEST_PASSWORD},
        )
        assert signup.status_code == 201, signup.text
        token = signup.json()["access_token"]
        client.headers["Authorization"] = f"Bearer {token}"
        yield client


@pytest.fixture
async def second_authed_client(opensearch_client: OpenSearch, auth_db_override):
    """Second authenticated user for isolation tests."""
    app.state.opensearch = opensearch_client
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        email = f"other-{uuid.uuid4().hex[:8]}@example.com"
        signup = await client.post(
            "/auth/signup",
            json={"email": email, "password": TEST_PASSWORD},
        )
        assert signup.status_code == 201, signup.text
        token = signup.json()["access_token"]
        client.headers["Authorization"] = f"Bearer {token}"
        yield client


@pytest.fixture
def unique_doc_id() -> str:
    return str(uuid.uuid4())
