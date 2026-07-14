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
        "RENAME",
        "ATTACH",
        "DETACH",
        "DO",
        "PREPARE",
        "DEALLOCATE",
        "LISTEN",
        "NOTIFY",
        "UNLISTEN",
        "LOAD",
        "REFRESH",
        "SECURITY",
    }
)

# Explicit DDL / schema-change verbs (belt-and-suspenders beyond sqlparse type).
_DDL_LEADING_RE = re.compile(
    r"^\s*(?:"
    r"CREATE|DROP|ALTER|TRUNCATE|RENAME|COMMENT\s+ON|"
    r"GRANT|REVOKE|REINDEX|CLUSTER|VACUUM|ANALYZE|"
    r"ATTACH|DETACH|REFRESH\s+MATERIALIZED\s+VIEW"
    r")\b",
    re.IGNORECASE,
)

_DDL_STATEMENT_TYPES = frozenset(
    {
        "CREATE",
        "DROP",
        "ALTER",
        "TRUNCATE",
        "GRANT",
        "REVOKE",
    }
)

_COMMENT_STRIP_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")


class SqlPolicyError(ValueError):
    """Raised when SQL text violates the read-only SELECT policy."""


def _strip_sql_comments(sql: str) -> str:
    without_block = _COMMENT_STRIP_RE.sub(" ", sql)
    return _LINE_COMMENT_RE.sub(" ", without_block)


def reject_ddl_sql(sql: str) -> None:
    """Reject DDL / schema-altering SQL even when a fuller allowlist is disabled."""
    cleaned = _strip_sql_comments((sql or "").strip())
    if not cleaned:
        return
    if _DDL_LEADING_RE.match(cleaned):
        raise SqlPolicyError("DDL / schema-changing SQL is not allowed")

    statements = [stmt for stmt in sqlparse.parse(cleaned) if str(stmt).strip()]
    for statement in statements:
        statement_type = (statement.get_type() or "").upper()
        if statement_type in _DDL_STATEMENT_TYPES:
            raise SqlPolicyError(f"DDL statement type is not allowed: {statement_type}")
        for token in statement.flatten():
            if not getattr(token, "is_keyword", False):
                continue
            keyword = str(token.value).upper()
            if keyword in {"CREATE", "DROP", "ALTER", "TRUNCATE", "RENAME"}:
                raise SqlPolicyError(f"forbidden DDL keyword: {keyword}")


def validate_sql_allowed(sql: str) -> None:
    """Allow exactly one read-only SELECT statement."""
    cleaned = _strip_sql_comments((sql or "").strip())
    if not cleaned:
        raise SqlPolicyError("empty SQL query")

    reject_ddl_sql(cleaned)

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
