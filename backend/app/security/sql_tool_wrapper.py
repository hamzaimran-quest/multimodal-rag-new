"""Wrap LangChain SQL tools with allowlist validation and audit logging."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any

from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.callbacks.manager import CallbackManagerForToolRun

from app.config import settings
from app.security.sql_audit import log_sql_audit
from app.security.sql_policy import (
    SqlPolicyError,
    reject_dangerous_functions,
    reject_ddl_sql,
    reject_system_catalog_access,
    validate_sql_allowed,
)

logger = logging.getLogger(__name__)

# QuerySQLDataBaseTool._run — verified against LangChain at wrap time.
_EXPECTED_RUN_PARAM_NAMES: tuple[str, ...] = ("query", "run_manager")


@dataclass(frozen=True)
class SqlAuditContext:
    user_id: int | None = None
    connection_label: str | None = None


def _sql_db_query_run_param_names() -> tuple[str, ...]:
    """Parameter names for QuerySQLDataBaseTool._run (excluding self)."""
    return tuple(
        name
        for name in inspect.signature(QuerySQLDataBaseTool._run).parameters
        if name != "self"
    )


def _assert_sql_db_query_tool(tool: Any) -> None:
    param_names = _sql_db_query_run_param_names()
    if param_names != _EXPECTED_RUN_PARAM_NAMES:
        raise RuntimeError(
            "QuerySQLDataBaseTool._run signature changed "
            f"(got {param_names!r}, expected {_EXPECTED_RUN_PARAM_NAMES!r}); "
            "update secure_sql_tools to match LangChain."
        )


def _query_from_run_call(*args: Any, **kwargs: Any) -> str:
    """Resolve query using the two conventions LangChain _to_args_and_kwargs emits."""
    if args:
        query = args[0]
    elif "query" in kwargs:
        query = kwargs["query"]
    else:
        raise RuntimeError("sql_db_query call is missing the query argument")
    if not isinstance(query, str):
        raise TypeError("sql_db_query query must be a string")
    return query


def _guard_sql_query(query: str, *, audit: SqlAuditContext | None) -> None:
    ctx = audit or SqlAuditContext()
    try:
        # Always block DDL, dangerous functions, and catalog reads even if the
        # full SELECT allowlist is disabled.
        reject_ddl_sql(query)
        reject_dangerous_functions(query)
        reject_system_catalog_access(query)
        if settings.security_sql_allowlist_enabled:
            validate_sql_allowed(query)
    except SqlPolicyError as exc:
        log_sql_audit(
            user_id=ctx.user_id,
            connection_label=ctx.connection_label,
            sql=query,
            allowed=False,
            reason=str(exc),
        )
        raise
    log_sql_audit(
        user_id=ctx.user_id,
        connection_label=ctx.connection_label,
        sql=query,
        allowed=True,
    )


def secure_sql_tools(tools: list[Any], *, audit: SqlAuditContext | None) -> list[Any]:
    """Return tools with sql_db_query execution gated by the SQL policy."""
    secured: list[Any] = []
    for tool in tools:
        if getattr(tool, "name", "") != "sql_db_query":
            secured.append(tool)
            continue

        _assert_sql_db_query_tool(tool)

        original_run = getattr(tool, "_run", None)
        original_arun = getattr(tool, "_arun", None)

        def _run(
            query: str,
            run_manager: CallbackManagerForToolRun | None = None,
            *,
            _original: Any = original_run,
        ) -> Any:
            _guard_sql_query(query, audit=audit)
            if _original is None:
                raise RuntimeError("sql_db_query tool is missing _run")
            result = _original(query, run_manager)
            result_text = str(result or "")
            logger.info(
                "SQL_TOOL sql_db_query result_chars=%s result_preview=%r query_preview=%r",
                len(result_text),
                " ".join(result_text.split())[:500],
                " ".join(query.split())[:300],
            )
            return result

        async def _arun(
            *args: Any,
            _original: Any = original_arun,
            **kwargs: Any,
        ) -> Any:
            query = _query_from_run_call(*args, **kwargs)
            _guard_sql_query(query, audit=audit)
            if _original is None:
                raise RuntimeError("sql_db_query tool is missing _arun")
            result = await _original(*args, **kwargs)
            result_text = str(result or "")
            logger.info(
                "SQL_TOOL sql_db_query_async result_chars=%s result_preview=%r query_preview=%r",
                len(result_text),
                " ".join(result_text.split())[:500],
                " ".join(query.split())[:300],
            )
            return result

        tool._run = _run  # type: ignore[method-assign]
        if original_arun is not None:
            tool._arun = _arun  # type: ignore[method-assign]
        secured.append(tool)
    return secured
