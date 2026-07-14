"""Standalone query rewriting for follow-ups — prior queries and assistant reply, aux LLM."""

from __future__ import annotations

import logging
import re

import httpx

from app.config import settings
from app.llm.groq import GROQ_CHAT_COMPLETIONS_URL
from app.retrieval.query_anchor import extract_named_phrases, merge_retrieval_anchor_phrases

logger = logging.getLogger(__name__)

QUERY_REWRITE_SYSTEM = """You rewrite the user's latest message into a standalone retrieval query.

You receive prior user questions from this chat (oldest first), optionally the most recent assistant reply, and the latest user message. Combine them only when the latest message depends on earlier turns; otherwise denoise the latest message.

## Goal

Strip noise only. Keep every important ask / segment from the latest message.

## Rules

- Output ONLY the standalone query text.
- No quotes, labels, or explanation.
- Remove filler only: please, can you, I'd like, polite framing, and duplicated words.
- If the latest message has multiple asks (joined by "and", "also", "as well as", etc.), keep ALL of them in the rewrite. Never collapse to a single topic.
- Do not drop metrics, years, segment/region names, people, roles, or narrative asks that appear in the latest message.
- Resolve pronouns, ellipsis, and vague references from prior context when needed (e.g. "compare that" → "compare Huawei revenue 2024 2025", "show her image" → "Meng Wanzhou rotating chairwoman portrait photo").
- Add only roles, titles, section labels, or entity names needed to disambiguate the current ask.
- Do NOT append full segment lists, region lists, product lines, cast lists, or other entity enumerations from prior turns unless the latest message explicitly asks about those entities.
- Do NOT copy long answer excerpts or multi-sentence summaries from the assistant reply into the query.
- Preserve specific names, numbers, and entities from prior questions or the latest assistant reply when resolving references.
- Keep full titles intact, including any text before a colon (e.g. "Name: Subtitle" must stay together).
- Do not drop distinctive named phrases that already appear in the latest message.
- When the latest message asks to see/show a visual (photo, portrait, image, figure, diagram), include short search-useful phrases that identify what to show. Omit filler and full sentences.
- If the latest message is already self-contained, prefer near-literal output (noise stripped only)."""

_MAX_REWRITE_CHARS = 500
_OVEREXPANSION_LENGTH_RATIO = 1.75
_OVEREXPANSION_EXTRA_CHARS = 50
_OVEREXPANSION_MIN_NEW_PROPER_NOUNS = 3
_TRIM_EXTRA_CHARS = 60

_CAPITALIZED_WORD_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
_VAGUE_FOLLOWUP_RE = re.compile(
    r"\b(?:that|this|it|them|those|these|him|her|his|hers|their|there)\b",
    re.IGNORECASE,
)
_VISUAL_FOLLOWUP_RE = re.compile(
    r"\b(?:show|display|see|image|photo|portrait|picture|figure|diagram)\b",
    re.IGNORECASE,
)
_COMPARE_INTENT_RE = re.compile(r"\b(?:compare|comparison|versus|vs)\b", re.IGNORECASE)
_YEAR_TOKEN_RE = re.compile(r"^(?:19|20)\d{2}$")
_MULTI_INTENT_RE = re.compile(
    r"\b(?:and also|as well as|and tell me|and what|plus)\b",
    re.IGNORECASE,
)
_METRIC_CUE_RE = re.compile(
    r"\b(?:revenue|growth|segment|segments|cagr|profit|total|count|average|sum)\b",
    re.IGNORECASE,
)
_NARRATIVE_CUE_RE = re.compile(
    r"\b(?:chairwoman|chairman|statement|stated|says|"
    r"message from|in the document|in the pdf|in the text|"
    r"from the document|from the pdf|from the text|"
    r"document|pdf|uploaded)\b",
    re.IGNORECASE,
)
_REWRITE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "also",
        "or",
        "for",
        "of",
        "to",
        "in",
        "on",
        "our",
        "my",
        "me",
        "i",
        "you",
        "please",
        "can",
        "could",
        "would",
        "tell",
        "show",
        "what",
        "about",
        "from",
        "with",
        "this",
        "that",
        "those",
        "these",
    }
)


def _proper_noun_signals(text: str) -> set[str]:
    signals = {phrase.casefold() for phrase in extract_named_phrases(text)}
    for match in _CAPITALIZED_WORD_RE.finditer(text):
        signals.add(match.group(0).casefold())
    for match in _ACRONYM_RE.finditer(text):
        signals.add(match.group(0).casefold())
    return signals


def _significant_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9']+", text.casefold())
        if len(token) > 2 and token not in _REWRITE_STOPWORDS
    ]


def _is_multi_intent_query(text: str) -> bool:
    if _MULTI_INTENT_RE.search(text):
        return True
    return bool(_METRIC_CUE_RE.search(text) and _NARRATIVE_CUE_RE.search(text))


def _preserve_multi_intent_segments(original: str, rewritten: str) -> str:
    """Fall back to original when a multi-intent rewrite drops key segments."""
    if not rewritten or not _is_multi_intent_query(original):
        return rewritten

    orig_tokens = _significant_tokens(original)
    if not orig_tokens:
        return rewritten

    rew_token_set = set(_significant_tokens(rewritten))
    kept = sum(1 for token in orig_tokens if token in rew_token_set)
    keep_ratio = kept / len(orig_tokens)

    metric_in_original = bool(_METRIC_CUE_RE.search(original))
    narrative_in_original = bool(_NARRATIVE_CUE_RE.search(original))
    metric_in_rewrite = bool(_METRIC_CUE_RE.search(rewritten))
    narrative_in_rewrite = bool(_NARRATIVE_CUE_RE.search(rewritten))
    lost_cue_family = (metric_in_original and not metric_in_rewrite) or (
        narrative_in_original and not narrative_in_rewrite
    )

    if keep_ratio < 0.6 or lost_cue_family:
        logger.info(
            "QUERY_REWRITE preserve_multi_intent keep_ratio=%.2f lost_cue_family=%s "
            "original=%r rewritten=%r",
            keep_ratio,
            lost_cue_family,
            original[:120],
            rewritten[:120],
        )
        return original
    return rewritten


def _is_visual_followup(original: str) -> bool:
    return bool(_VISUAL_FOLLOWUP_RE.search(original))


def _has_compare_intent(original: str, rewritten: str) -> bool:
    return bool(
        _COMPARE_INTENT_RE.search(original)
        or rewritten.casefold().startswith("compare")
    )


def _looks_like_entity_enumeration(
    rewritten: str,
    *,
    new_signals: set[str],
    original: str | None = None,
) -> bool:
    if original and _is_visual_followup(original):
        return False

    comma_segments = [part.strip() for part in rewritten.split(",") if part.strip()]
    if len(comma_segments) >= 4:
        return True

    word_count = len(rewritten.split())
    if len(new_signals) >= 8 and word_count >= 12:
        return True

    # Repeated business-line style tokens often indicate pasted segment lists.
    business_like = sum(
        1
        for token in rewritten.split()
        if token.casefold() in {"business", "segment", "segments", "division", "divisions", "unit", "units"}
    )
    return business_like >= 3


def _is_allowed_pronoun_resolution(
    original: str,
    rewritten: str,
    *,
    new_signals: set[str],
) -> bool:
    """Short follow-ups may grow while resolving a single referent."""
    if not _VAGUE_FOLLOWUP_RE.search(original):
        return False
    if "," in rewritten:
        return False
    if _looks_like_entity_enumeration(rewritten, new_signals=new_signals, original=original):
        return False
    return len(rewritten.split()) <= 12


def _limit_vague_followup_rewrite(original: str, rewritten: str) -> str:
    words = rewritten.split()
    if _has_compare_intent(original, rewritten):
        last_year_index = -1
        for index, word in enumerate(words):
            if _YEAR_TOKEN_RE.fullmatch(word):
                last_year_index = index
        if last_year_index >= 0:
            return " ".join(words[: last_year_index + 1])
        return " ".join(words[:6])

    if _is_visual_followup(original):
        return " ".join(words[:12])

    return " ".join(words[:10])


def _rewrite_over_expanded(original: str, rewritten: str) -> bool:
    """True when rewrite likely dumped prior-turn entity lists into the query."""
    if not rewritten or rewritten.casefold() == original.casefold():
        return False

    new_signals = _proper_noun_signals(rewritten) - _proper_noun_signals(original)
    if _is_allowed_pronoun_resolution(original, rewritten, new_signals=new_signals):
        return False

    orig_len = len(original)
    rew_len = len(rewritten)
    if rew_len <= orig_len + _OVEREXPANSION_EXTRA_CHARS:
        return False
    if rew_len < max(orig_len + 1, 1) * _OVEREXPANSION_LENGTH_RATIO:
        return False

    if not _looks_like_entity_enumeration(rewritten, new_signals=new_signals, original=original):
        return False

    return len(new_signals) >= _OVEREXPANSION_MIN_NEW_PROPER_NOUNS


def _trim_over_expanded_rewrite(original: str, rewritten: str) -> str:
    """Prefer a shorter rewrite when the model appended wholesale entity lists."""
    if not _rewrite_over_expanded(original, rewritten):
        return rewritten

    if _VAGUE_FOLLOWUP_RE.search(original):
        word_limited = _limit_vague_followup_rewrite(original, rewritten)
        if word_limited and not _rewrite_over_expanded(original, word_limited):
            logger.info(
                "QUERY_REWRITE trimmed over_expansion original_len=%s rewritten_len=%s trimmed_len=%s",
                len(original),
                len(rewritten),
                len(word_limited),
            )
            return word_limited

    max_len = max(len(original) + _TRIM_EXTRA_CHARS, int(len(original) * 1.4))
    if len(rewritten) > max_len:
        trimmed = rewritten[:max_len].rsplit(" ", 1)[0].strip()
    else:
        trimmed = rewritten

    if trimmed and not _rewrite_over_expanded(original, trimmed):
        logger.info(
            "QUERY_REWRITE trimmed over_expansion original_len=%s rewritten_len=%s trimmed_len=%s",
            len(original),
            len(rewritten),
            len(trimmed),
        )
        return trimmed

    logger.info(
        "QUERY_REWRITE fallback_original over_expansion original_len=%s rewritten_len=%s",
        len(original),
        len(rewritten),
    )
    return original


def _build_rewrite_user_prompt(
    prior_queries: list[str],
    user_query: str,
    last_assistant_reply: str | None = None,
) -> str:
    lines: list[str] = []
    if prior_queries:
        lines.append("Prior user questions:")
        for index, question in enumerate(prior_queries, start=1):
            lines.append(f"{index}. {question}")
        lines.append("")
    if last_assistant_reply:
        lines.append("Latest assistant reply:")
        lines.append(last_assistant_reply)
        lines.append("")
    lines.append(f"Latest message: {user_query}")
    return "\n".join(lines)


async def rewrite_query_for_retrieval(
    user_query: str,
    prior_queries: list[str] | None,
    last_assistant_reply: str | None = None,
) -> str:
    """
    Produce a standalone retrieval query from the latest message and prior chat context.

    Skips the LLM when rewrite is disabled or there is no prior user question or reply.
    """
    cleaned_query = user_query.strip()
    if not cleaned_query:
        return user_query

    prior = [q.strip() for q in (prior_queries or []) if q and q.strip()]
    last_reply = (last_assistant_reply or "").strip() or None
    if not settings.query_rewrite_enabled or (not prior and not last_reply):
        return user_query

    if not settings.groq_configured:
        return user_query

    messages = [
        {"role": "system", "content": QUERY_REWRITE_SYSTEM},
        {
            "role": "user",
            "content": _build_rewrite_user_prompt(prior, cleaned_query, last_reply),
        },
    ]
    payload = {
        "model": settings.query_rewrite_model,
        "messages": messages,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(GROQ_CHAT_COMPLETIONS_URL, headers=headers, json=payload)
            response.raise_for_status()
            rewritten = (response.json()["choices"][0]["message"].get("content") or "").strip()
            rewritten = rewritten.strip("\"'")
            if rewritten and len(rewritten) <= _MAX_REWRITE_CHARS:
                sanitized = _trim_over_expanded_rewrite(cleaned_query, rewritten)
                sanitized = _preserve_multi_intent_segments(cleaned_query, sanitized)
                merged = merge_retrieval_anchor_phrases(
                    sanitized,
                    fallback_queries=[cleaned_query],
                )
                logger.info(
                    "QUERY_REWRITE model=%s prior_count=%s original=%r rewritten=%r sanitized=%r merged=%r",
                    settings.query_rewrite_model,
                    len(prior),
                    cleaned_query[:80],
                    rewritten[:80],
                    sanitized[:80],
                    merged[:80],
                )
                return merged
    except Exception:
        logger.warning(
            "QUERY_REWRITE failed model=%s query=%r",
            settings.query_rewrite_model,
            cleaned_query[:80],
            exc_info=True,
        )
    return user_query
