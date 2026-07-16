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

# Server-side functions that read/write files, open network connections, or
# control other backends — none of these are SQL keywords, so a plain SELECT
# calling them would otherwise pass the keyword-based allowlist untouched.
_FORBIDDEN_FUNCTIONS = frozenset(
    {
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_ls_logdir",
        "pg_ls_waldir",
        "pg_ls_tmpdir",
        "pg_ls_archive_statusdir",
        "pg_stat_file",
        "lo_import",
        "lo_export",
        "lo_read",
        "lo_write",
        "lo_open",
        "lo_creat",
        "lo_create",
        "dblink",
        "dblink_connect",
        "dblink_connect_u",
        "dblink_exec",
        "dblink_send_query",
        "dblink_open",
        "pg_terminate_backend",
        "pg_cancel_backend",
        "pg_reload_conf",
        "pg_rotate_logfile",
        "pg_sleep",
        "pg_sleep_for",
        "pg_sleep_until",
    }
)

_FUNCTION_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# Schema/catalog metadata is a disclosure concern handled at the product level
# (chat should not dump raw DB structure) — block it at the SQL layer too so
# it doesn't depend solely on classifying the user's phrasing.
_SYSTEM_CATALOG_RE = re.compile(r"\b(?:pg_catalog|information_schema)\s*\.", re.IGNORECASE)


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


def reject_dangerous_functions(sql: str) -> None:
    """Reject calls to server-side file/network/backend-control functions,
    even inside an otherwise-valid single SELECT statement."""
    cleaned = _strip_sql_comments((sql or "").strip())
    if not cleaned:
        return
    for match in _FUNCTION_CALL_RE.finditer(cleaned):
        name = match.group(1).lower()
        if name in _FORBIDDEN_FUNCTIONS:
            raise SqlPolicyError(f"forbidden SQL function call: {name}")


def reject_system_catalog_access(sql: str) -> None:
    """Reject direct reads of information_schema / pg_catalog metadata."""
    cleaned = _strip_sql_comments((sql or "").strip())
    if not cleaned:
        return
    if _SYSTEM_CATALOG_RE.search(cleaned):
        raise SqlPolicyError("access to information_schema/pg_catalog is not allowed")


def validate_sql_allowed(sql: str) -> None:
    """Allow exactly one read-only SELECT statement."""
    cleaned = _strip_sql_comments((sql or "").strip())
    if not cleaned:
        raise SqlPolicyError("empty SQL query")

    reject_ddl_sql(cleaned)
    reject_dangerous_functions(cleaned)
    reject_system_catalog_access(cleaned)

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
