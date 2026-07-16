"""Screen assistant output before it is streamed or persisted."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import settings

_CONNECTION_STRING_RE = re.compile(
    r"\b(?:postgres(?:ql)?|mysql|mariadb|mssql)://\S+",
    re.IGNORECASE,
)
_API_KEY_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{10,}|Bearer\s+[A-Za-z0-9._-]{10,})\b"
)
_SQL_STATEMENT_RE = re.compile(
    r"\b(?:SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT)\b.{0,200}?\bFROM\b",
    re.IGNORECASE | re.DOTALL,
)
_SQL_CODE_FENCE_RE = re.compile(r"```\s*sql\b", re.IGNORECASE)
_STACK_TRACE_RE = re.compile(r"Traceback \(most recent call last\):", re.IGNORECASE)
_ENV_LEAK_RE = re.compile(r"\b(?:GROQ_API_KEY|SQL_AGENT_OPENROUTER_API_KEY|JWT_SECRET)\b")
_SYSTEM_PROMPT_LEAK_RE = re.compile(
    r"You are the routing assistant for a document Q&A product",
    re.IGNORECASE,
)

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)

# Cheap prefix-only version of the checks above, used to decide when a live
# token stream needs to pause and buffer rather than release immediately.
# Matching here does not mean the text is blocked — only that it might be the
# start of a pattern scan_output_text checks for, so the caller should hold
# it back until the full pattern either completes (block) or the stream ends.
_STREAM_RISK_ANCHOR_RE = re.compile(
    r"postgres(?:ql)?://|mysql://|mariadb://|mssql://"
    r"|sk-[A-Za-z0-9_-]"
    r"|Bearer\s"
    r"|Traceback \(most recent call last\):"
    r"|GROQ_API_KEY|SQL_AGENT_OPENROUTER_API_KEY|JWT_SECRET"
    r"|You are the routing assistant for a document Q&A product"
    r"|```"
    r"|\b(?:SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT)\b",
    re.IGNORECASE,
)


def earliest_risk_anchor_offset(text: str) -> int | None:
    """Offset of the earliest substring in `text` that could be the start of
    a pattern scan_output_text checks for, or None if nothing looks risky."""
    if not settings.security_output_guard_enabled:
        return None
    match = _STREAM_RISK_ANCHOR_RE.search(text)
    return match.start() if match else None


@dataclass(frozen=True)
class OutputVerdict:
    allowed: bool
    reason: str | None = None
    safe_message: str | None = None


_SAFE_BLOCKED_MESSAGE = (
    "I can't show that response because it may contain sensitive internal or personal data. "
    "Please rephrase your question."
)


def scan_output_text(text: str) -> OutputVerdict:
    """Return whether assistant output is safe to show the user."""
    if not settings.security_output_guard_enabled:
        return OutputVerdict(allowed=True)

    body = text or ""
    if not body.strip():
        return OutputVerdict(allowed=True)

    checks: list[tuple[re.Pattern[str], str]] = [
        (_CONNECTION_STRING_RE, "connection_string"),
        (_API_KEY_RE, "api_key"),
        (_STACK_TRACE_RE, "stack_trace"),
        (_ENV_LEAK_RE, "env_secret_name"),
        (_SYSTEM_PROMPT_LEAK_RE, "system_prompt_leak"),
        (_SQL_STATEMENT_RE, "raw_sql"),
        (_SQL_CODE_FENCE_RE, "raw_sql"),
    ]
    for pattern, reason in checks:
        if pattern.search(body):
            return OutputVerdict(
                allowed=False,
                reason=reason,
                safe_message=_SAFE_BLOCKED_MESSAGE,
            )

    if len(_EMAIL_RE.findall(body)) >= 3:
        return OutputVerdict(
            allowed=False,
            reason="possible_pii",
            safe_message=_SAFE_BLOCKED_MESSAGE,
        )

    return OutputVerdict(allowed=True)
