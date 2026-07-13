"""API tests for SQL Agent connection management."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import AsyncClient

from tests.test_auth import auth_client


async def _signup_and_token(client: AsyncClient, email: str) -> str:
    response = await client.post("/auth/signup", json={"email": email, "password": "supersecret1"})
    assert response.status_code in {200, 201}
    return response.json()["access_token"]


@patch("app.sql_agent.service._refresh_schema_cache")
@patch("app.api.sql_agent.sql_service.test_postgres_connection")
async def test_sql_connection_lifecycle(mock_test, mock_warm, auth_client: AsyncClient) -> None:
    mock_test.return_value = None
    token_a = await _signup_and_token(auth_client, "sql-a@example.com")
    headers = {"Authorization": f"Bearer {token_a}"}

    status = await auth_client.get("/sql-agent/status", headers=headers)
    assert status.status_code == 200
    assert status.json()["has_active"] is False

    created = await auth_client.post(
        "/sql-agent/connections",
        headers=headers,
        json={
            "connection_url": "postgresql://user:pass@localhost:5432/db1",
            "display_name": "Primary",
            "description": "Orders database",
        },
    )
    assert created.status_code == 201
    conn_id = created.json()["id"]
    assert created.json()["is_active"] is True

    second = await auth_client.post(
        "/sql-agent/connections",
        headers=headers,
        json={
            "connection_url": "postgresql://user:pass@localhost:5432/db2",
            "display_name": "Secondary",
            "description": "Warehouse",
            "activate": False,
        },
    )
    assert second.status_code == 201
    second_id = second.json()["id"]
    assert second.json()["is_active"] is False

    activated = await auth_client.post(
        f"/sql-agent/connections/{second_id}/activate",
        headers=headers,
    )
    assert activated.status_code == 200
    assert activated.json()["is_active"] is True

    status = await auth_client.get("/sql-agent/status", headers=headers)
    payload = status.json()
    assert payload["has_active"] is True
    assert payload["active_connection"]["id"] == second_id
    active_rows = [row for row in payload["connections"] if row["is_active"]]
    assert len(active_rows) == 1

    await auth_client.post("/sql-agent/deactivate", headers=headers)
    status = await auth_client.get("/sql-agent/status", headers=headers)
    assert status.json()["has_active"] is False

    deleted = await auth_client.delete(f"/sql-agent/connections/{conn_id}", headers=headers)
    assert deleted.status_code == 204


@patch("app.sql_agent.service._refresh_schema_cache")
@patch("app.api.sql_agent.sql_service.test_postgres_connection")
async def test_sql_connection_isolated_between_users(mock_test, mock_warm, auth_client: AsyncClient) -> None:
    mock_test.return_value = None
    token_a = await _signup_and_token(auth_client, "sql-owner@example.com")
    token_b = await _signup_and_token(auth_client, "sql-other@example.com")

    created = await auth_client.post(
        "/sql-agent/connections",
        headers={"Authorization": f"Bearer {token_a}"},
        json={
            "connection_url": "postgresql://user:pass@localhost:5432/db1",
            "display_name": "Owner DB",
            "description": "Private",
        },
    )
    conn_id = created.json()["id"]

    forbidden = await auth_client.post(
        f"/sql-agent/connections/{conn_id}/activate",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert forbidden.status_code == 404
