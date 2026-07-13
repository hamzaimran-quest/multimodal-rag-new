"""Hard allowlist for SQL executed by the agent."""

from __future__ import annotations

import re

import sqlparse

_FORBIDDEN_KEYWORDS = frozenset(
    {
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "TRUNCATE",
        "CREATE",
        "GRANT",
        "REVOKE",
        "COPY",
        "EXECUTE",
        "CALL",
        "MERGE",
        "REPLACE",
        "VACUUM",
        "ANALYZE",
        "COMMENT",
        "REINDEX",
        "CLUSTER",
        "DISCARD",
        "RESET",
        "SET",
        "BEGIN",
        "COMMIT",
        "ROLLBACK",
        "SAVEPOINT",
        "LOCK",
        "UNLOCK",
    }
)

_COMMENT_STRIP_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")


class SqlPolicyError(ValueError):
    """Raised when SQL text violates the read-only SELECT policy."""


def _strip_sql_comments(sql: str) -> str:
    without_block = _COMMENT_STRIP_RE.sub(" ", sql)
    return _LINE_COMMENT_RE.sub(" ", without_block)


def validate_sql_allowed(sql: str) -> None:
    """Allow exactly one read-only SELECT statement."""
    cleaned = _strip_sql_comments((sql or "").strip())
    if not cleaned:
        raise SqlPolicyError("empty SQL query")

    statements = [stmt for stmt in sqlparse.parse(cleaned) if str(stmt).strip()]
    if len(statements) != 1:
        raise SqlPolicyError("only a single SQL statement is allowed")

    statement = statements[0]
    statement_type = (statement.get_type() or "").upper()
    if statement_type != "SELECT":
        raise SqlPolicyError(f"only SELECT statements are allowed (got {statement_type or 'UNKNOWN'})")

    for token in statement.flatten():
        if getattr(token, "is_keyword", False) and str(token.value).upper() in _FORBIDDEN_KEYWORDS:
            raise SqlPolicyError(f"forbidden SQL keyword: {token.value.upper()}")

    if re.search(r";\s*\S", cleaned):
        raise SqlPolicyError("multi-statement SQL is not allowed")
