"""Resolve whether the user restricted answers to documents, database, or both."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum

import httpx

from app.config import settings
from app.llm.groq import GROQ_CHAT_COMPLETIONS_URL

logger = logging.getLogger(__name__)


class SourceIntent(str, Enum):
    DOCS_ONLY = "docs_only"
    DB_ONLY = "db_only"
    BOTH = "both"
    AMBIGUOUS = "ambiguous"


class SourceIntentMethod(str, Enum):
    REGEX = "regex"
    LLM = "llm"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class SourceIntentResult:
    intent: SourceIntent
    method: SourceIntentMethod
    reason: str = ""


# High-precision only: clear source restrictions. Weak/ambiguous phrasing goes to the LLM.
_DB_ONLY_SOURCE_RE = re.compile(
    r"\b(?:"
    r"from the (?:db|database)|in the (?:db|database)|"
    r"(?:db|database) only|sql only|only (?:the )?(?:db|database)|"
    r"ignore (?:the )?documents?|don't use (?:the )?documents?|"
    r"don't (?:search|use) (?:the )?(?:docs?|documents?)"
    r")\b",
    re.IGNORECASE,
)
_DOCS_ONLY_SOURCE_RE = re.compile(
    r"\b(?:"
    r"from the (?:doc|docs|document|pdf|upload|file|spreadsheet|excerpt)s?|"
    r"in the (?:doc|docs|document|pdf|upload|file|spreadsheet|excerpt)s?|"
    r"(?:doc|docs|document|pdf|upload|file|spreadsheet|excerpt)s? only|"
    r"only (?:the )?(?:doc|docs|document|pdf|upload|file|spreadsheet|excerpt)s?|"
    r"ignore (?:the )?(?:db|database)|don't use (?:the )?(?:db|database)|"
    r"don't (?:query|use) (?:the )?(?:db|database)"
    r")\b",
    re.IGNORECASE,
)

_SOURCE_INTENT_SYSTEM = """You classify whether the user restricted which data source to use.

Sources available in this product:
- uploaded documents (PDF/DOCX/XLSX — "doc", "docs", "PDF", "file", "upload", "excerpt")
- a connected PostgreSQL database ("database", "DB", "SQL")

Return ONLY valid JSON:
{"intent":"docs_only"|"db_only"|"both"|"ambiguous","reason":"short phrase"}

Rules:
- docs_only: user clearly wants documents/PDF/uploads only (e.g. "from the doc", "in the excerpt", "PDFs only", "ignore the database")
- db_only: user clearly wants the database/SQL only (e.g. "from the database", "SQL only", "ignore documents")
- both: user explicitly asks to use documents AND database together
- ambiguous: no clear source restriction (normal factual questions default here)

Do NOT classify based on whether the topic sounds numerical or document-like — only on source preference language.
When unsure, use ambiguous."""

_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_VALID_INTENTS = frozenset(item.value for item in SourceIntent)


def match_source_intent_regex(*queries: str) -> SourceIntent | None:
    """Return a clear docs/db-only match, or None if phrasing is not high-confidence."""
    texts = [(q or "").strip() for q in queries if (q or "").strip()]
    if not texts:
        return None

    docs = any(_DOCS_ONLY_SOURCE_RE.search(t) for t in texts)
    db = any(_DB_ONLY_SOURCE_RE.search(t) for t in texts)
    if docs and db:
        return None
    if docs:
        return SourceIntent.DOCS_ONLY
    if db:
        return SourceIntent.DB_ONLY
    return None


def user_restricted_to_database_only(*queries: str) -> bool:
    return match_source_intent_regex(*queries) == SourceIntent.DB_ONLY


def user_restricted_to_documents_only(*queries: str) -> bool:
    return match_source_intent_regex(*queries) == SourceIntent.DOCS_ONLY


def _parse_llm_intent(raw: str) -> tuple[SourceIntent, str] | None:
    text = (raw or "").strip()
    if not text:
        return None

    candidates = [text]
    match = _JSON_OBJECT_RE.search(text)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        intent_raw = str(payload.get("intent") or "").strip().lower()
        if intent_raw in _VALID_INTENTS:
            reason = str(payload.get("reason") or "").strip()[:120]
            return SourceIntent(intent_raw), reason
    return None


def _build_classifier_user_prompt(user_query: str, router_query: str) -> str:
    parts = [f"User message:\n{user_query.strip()}"]
    rewritten = (router_query or "").strip()
    if rewritten and rewritten != user_query.strip():
        parts.append(f"\nStandalone rewrite (context only):\n{rewritten}")
    return "\n".join(parts)


async def _classify_source_intent_with_llm(
    user_query: str,
    router_query: str,
) -> SourceIntentResult | None:
    if not settings.source_intent_classifier_enabled or not settings.groq_configured:
        return None

    payload = {
        "model": settings.resolved_source_intent_classifier_model,
        "messages": [
            {"role": "system", "content": _SOURCE_INTENT_SYSTEM},
            {
                "role": "user",
                "content": _build_classifier_user_prompt(user_query, router_query),
            },
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(
            timeout=settings.source_intent_classifier_timeout_seconds
        ) as client:
            response = await client.post(GROQ_CHAT_COMPLETIONS_URL, headers=headers, json=payload)
            response.raise_for_status()
            content = (response.json()["choices"][0]["message"].get("content") or "").strip()
    except Exception as exc:
        logger.warning("SOURCE_INTENT llm failed: %s", exc)
        return None

    parsed = _parse_llm_intent(content)
    if parsed is None:
        logger.warning(
            "SOURCE_INTENT llm unparseable content_preview=%r",
            content[:200],
        )
        return None

    intent, reason = parsed
    return SourceIntentResult(intent=intent, method=SourceIntentMethod.LLM, reason=reason)


async def resolve_source_intent(
    user_query: str,
    router_query: str | None = None,
) -> SourceIntentResult:
    """
    Detect explicit source preference for one turn.

    Order: high-precision regex → aux LLM → ambiguous fallback.
    """
    rewritten = (router_query or user_query or "").strip()
    original = (user_query or "").strip()

    regex_match = match_source_intent_regex(original, rewritten)
    if regex_match is not None:
        return SourceIntentResult(
            intent=regex_match,
            method=SourceIntentMethod.REGEX,
            reason="high_precision_phrase",
        )

    llm_result = await _classify_source_intent_with_llm(original, rewritten)
    if llm_result is not None:
        return llm_result

    return SourceIntentResult(
        intent=SourceIntent.AMBIGUOUS,
        method=SourceIntentMethod.FALLBACK,
        reason="classifier_unavailable_or_failed",
    )


def sql_allowed_for_intent(intent: SourceIntent, *, sql_active: bool) -> bool:
    if not sql_active:
        return False
    return intent in {SourceIntent.DB_ONLY, SourceIntent.BOTH, SourceIntent.AMBIGUOUS}


def docs_allowed_for_intent(intent: SourceIntent) -> bool:
    return intent in {SourceIntent.DOCS_ONLY, SourceIntent.BOTH, SourceIntent.AMBIGUOUS}
