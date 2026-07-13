"""Screen raw user messages before routing, rewrite, or tool execution."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import settings

_LEET_TRANSLATION = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "@": "a",
        "$": "s",
    }
)

_JAILBREAK_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bignore\b.{0,40}\b(instructions?|rules|policy|prompt|guidelines)\b",
        r"\bignore\b.{0,20}\b(all|any|previous|prior|earlier|my)\b.{0,20}\b(instructions?|rules|prompts?)\b",
        r"\b(disregard|forget|override|bypass|disable)\b.{0,30}\b(instructions|rules|system prompt|guardrails|safety|restrictions|content filter)\b",
        r"\b(pretend|act)\b.{0,20}\b(unrestricted|uncensored|jailbroken|without rules|without limits)\b",
        r"\b(reveal|expose|leak|output|print|show|repeat|dump|recite)\b.{0,30}\b(system prompt|hidden prompt|instructions|developer message|hidden instructions|initial prompt|original prompt)\b",
        r"\bwhat are your (instructions|rules|system prompt|guidelines)\b",
        r"\b(bypass|disable|turn off|remove)\b.{0,20}\b(guardrails|safety|content filter|restrictions)\b",
        r"\b(developer|sudo|god|jailbreak|dan)\s+mode\b",
        r"\byou are now\b.{0,30}\b(unrestricted|unfiltered|developer mode|dan)\b",
        r"\bact as (if )?you (have|had) no (rules|restrictions|limits|guidelines)\b",
        r"\bfrom now on\b.{0,30}\b(ignore|disregard|forget)\b",
        r"\bnew (system )?prompt\b",
        r"\bsystem override\b",
        r"\bend of user message\b",
        r"\[/system\]|\[system\]|\</user\>|\<system\>",
        r"\bdo anything now\b",
        r"\bDAN\b",
        r"\booc:\s",
        r"\brepeat (everything|the text|the message) above\b",
    )
)

_DESTRUCTIVE_SQL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bdrop\s+(table|schema|database)\b",
        r"\bdelete\s+from\b",
        r"\btruncate\s+(table\s+)?\w+",
        r"\balter\s+(table|database|schema)\b",
        r"\bupdate\s+\w+\s+set\b",
        r"\binsert\s+into\b",
        r"\bmerge\s+into\b",
        r"\breplace\s+into\b",
        r"\bcreate\s+(table|database|user|role)\b",
        r"\bgrant\b.{0,30}\b(to|on)\b",
        r"\brevoke\b.{0,30}\b(from|on)\b",
        r"\bcopy\b.{0,40}\bto\b",
        r"\bexecute\b.{0,20}\b(immediate|procedure|function)\b",
        r"\bexec\b\s*\(",
        r";\s*(drop|delete|truncate|alter|insert|update|create|grant|revoke)\b",
        r"\bpassword\s+hash(es)?\b",
        r"\beveryone'?s\b.{0,20}\bpassword\b",
        r"\bpg_sleep\s*\(",
        r"\bunion\b.{0,40}\bselect\b.{0,40}\bfrom\b",
    )
)

_SQL_QUERY_EXFIL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(show|print|reveal|display|give|tell|share|repeat|dump|output)\b.{0,25}\b(the\s+)?(sql\s+)?(query|queries|statement|statements)\b.{0,20}\b(you\s+)?(ran|used|executed|wrote)\b",
        r"\bwhat (sql\s+)?(query|queries) did you (run|use|execute|write)\b",
        r"\bwhat (sql\s+)?(query|queries) (was|were) (run|used|executed|written)\b",
        r"\bgive me the (sql\s+)?(select|query|statement)\b",
        r"\bshow me the (sql\s+)?(select|query|statement)\b",
        r"\b(reveal|print|show|repeat|dump)\b.{0,20}\bthe (sql\s+)?(select|query|statement)\b",
        r"\brepeat the (sql\s+)?(query|statement) (you\s+)?(ran|used|executed)\b",
        r"\bshare (your|the) (internal\s+)?(sql\s+)?(query|queries)\b",
    )
)

_SCHEMA_EXFIL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(show|list|describe|give|tell|display|dump|print|reveal)\b.{0,25}\b(the\s+)?(db|database)\s+schema\b",
        r"\b(show|list|describe|give|tell|display|dump|print|reveal)\b.{0,25}\b(the\s+)?(schema|table structure|column names|columns and types)\b",
        r"\bwhat tables\b.{0,30}\b(in the\s+)?(database|db)\b",
        r"\blist\b.{0,20}\b(all\s+)?tables\b",
        r"\bdescribe\s+(the\s+)?(database|schema|tables)\b",
        r"\binformation_schema\b",
    )
)

_TOXIC_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(kill yourself|kys)\b",
    )
)


@dataclass(frozen=True)
class InputVerdict:
    allowed: bool
    reason: str | None = None
    user_message: str | None = None


_BLOCKED_USER_MESSAGE = (
    "I can't help with that request. Ask a question about your uploaded documents "
    "or connected database using read-only queries."
)


def blocked_user_message() -> str:
    return _BLOCKED_USER_MESSAGE


def _normalize_for_screening(text: str) -> str:
    """Fold leetspeak and delimiter noise so simple obfuscations still match."""
    normalized = text.casefold().translate(_LEET_TRANSLATION)
    normalized = re.sub(r"[_\-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _matches_any(patterns: tuple[re.Pattern[str], ...], *candidates: str) -> bool:
    for candidate in candidates:
        if not candidate:
            continue
        for pattern in patterns:
            if pattern.search(candidate):
                return True
    return False


def classify_user_input(query: str) -> InputVerdict:
    """Fast rule-based screen of the raw user message."""
    if not settings.security_input_guard_enabled:
        return InputVerdict(allowed=True)

    text = (query or "").strip()
    if not text:
        return InputVerdict(allowed=True)

    normalized = _normalize_for_screening(text)

    if _matches_any(_TOXIC_PATTERNS, text, normalized):
        return InputVerdict(
            allowed=False,
            reason="abuse",
            user_message=_BLOCKED_USER_MESSAGE,
        )

    if _matches_any(_JAILBREAK_PATTERNS, text, normalized):
        return InputVerdict(
            allowed=False,
            reason="jailbreak",
            user_message=_BLOCKED_USER_MESSAGE,
        )

    if _matches_any(_DESTRUCTIVE_SQL_PATTERNS, text, normalized):
        return InputVerdict(
            allowed=False,
            reason="destructive_sql_request",
            user_message=_BLOCKED_USER_MESSAGE,
        )

    if _matches_any(_SQL_QUERY_EXFIL_PATTERNS, text, normalized):
        return InputVerdict(
            allowed=False,
            reason="sql_query_request",
            user_message=_BLOCKED_USER_MESSAGE,
        )

    if _matches_any(_SCHEMA_EXFIL_PATTERNS, text, normalized):
        return InputVerdict(
            allowed=False,
            reason="schema_request",
            user_message=_BLOCKED_USER_MESSAGE,
        )

    return InputVerdict(allowed=True)


async def classify_user_input_async(query: str) -> InputVerdict:
    """Rule-based screen first, then optional LLM fallback for nuanced policy checks."""
    verdict = classify_user_input(query)
    if not verdict.allowed:
        return verdict

    from app.security.input_llm_classifier import classify_user_input_with_llm

    llm_verdict = await classify_user_input_with_llm(query)
    if llm_verdict is None:
        return verdict
    return llm_verdict
