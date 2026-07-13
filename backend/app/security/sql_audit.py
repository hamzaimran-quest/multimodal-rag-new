"""Structured audit logging for SQL execution attempts."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("security.sql_audit")


def log_sql_audit(
    *,
    user_id: int | None,
    connection_label: str | None,
    sql: str,
    allowed: bool,
    reason: str | None = None,
) -> None:
    preview = " ".join((sql or "").split())[:500]
    payload: dict[str, Any] = {
        "user_id": user_id,
        "connection": connection_label,
        "allowed": allowed,
        "sql_preview": preview,
    }
    if reason:
        payload["reason"] = reason
    if allowed:
        logger.info("SQL_AUDIT allowed=%s", payload)
    else:
        logger.warning("SQL_AUDIT allowed=%s", payload)
