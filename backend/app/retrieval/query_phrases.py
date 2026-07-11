"""Extract multi-word lookup phrases for row and entity matching."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.retrieval.query_anchor import extract_named_phrases

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "of",
        "on",
        "or",
        "tell",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "who",
        "why",
        "with",
    }
)

_LOOKUP_MARKERS = (
    "about the movie ",
    "about the film ",
    "about the show ",
    "about the series ",
    "for the movie ",
    "for the film ",
    "for the show ",
    "for the series ",
    "about ",
    "for ",
    "titled ",
    "called ",
    "named ",
)

_LEADING_FILLERS = (
    "the ",
    "a ",
    "an ",
    "movie ",
    "film ",
    "show ",
    "series ",
)


def _raw_tokens(query: str) -> list[str]:
    return _TOKEN_RE.findall(query or "")


def _is_distinctive_token(token: str) -> bool:
    folded = token.casefold()
    if any(character.isdigit() for character in folded):
        return True
    if len(folded) >= 4 and folded not in _STOP_WORDS:
        return True
    if len(folded) >= 2 and folded not in _STOP_WORDS:
        return True
    return False


def _has_title_short_or_digit(token: str) -> bool:
    folded = token.casefold()
    if any(character.isdigit() for character in folded):
        return True
    return len(folded) <= 2 and folded not in _STOP_WORDS and folded.isalpha()


def _span_is_lookup_phrase(span: list[str]) -> bool:
    if len(span) < 2:
        return False
    distinctive = [token for token in span if _is_distinctive_token(token)]
    if not distinctive:
        return False
    if any(_has_title_short_or_digit(token) for token in span):
        return True
    return len(span) >= 2 and len(distinctive) >= 2


def _alphanumeric_spans(query: str) -> list[str]:
    tokens = _raw_tokens(query)
    phrases: list[str] = []
    max_span = 6
    for start in range(len(tokens)):
        if not (_is_distinctive_token(tokens[start]) or _has_title_short_or_digit(tokens[start])):
            continue
        for end in range(start + 2, min(len(tokens), start + max_span) + 1):
            span = tokens[start:end]
            if _span_is_lookup_phrase(span):
                phrases.append(" ".join(span))
    return phrases


def _trim_lookup_tail(tail: str) -> str:
    text = tail.strip()
    for separator in ("?", ".", "!", ",", ";", "\n"):
        if separator in text:
            text = text.split(separator, 1)[0].strip()
    changed = True
    while changed and text:
        changed = False
        lowered = text.casefold()
        for filler in _LEADING_FILLERS:
            if lowered.startswith(filler):
                text = text[len(filler) :].strip()
                changed = True
                break
    words = text.split()
    if not words:
        return ""
    return " ".join(words[:6]).strip()


def _phrase_after_markers(query: str) -> list[str]:
    folded = query.casefold()
    phrases: list[str] = []
    for marker in _LOOKUP_MARKERS:
        start = 0
        while True:
            index = folded.find(marker, start)
            if index < 0:
                break
            tail = query[index + len(marker) :]
            phrase = _trim_lookup_tail(tail)
            if phrase:
                phrases.append(phrase)
            start = index + len(marker)
    return phrases


def extract_lookup_phrases(query: str) -> list[str]:
    """Return distinctive phrases (longest first) for substring row/entity matching."""
    if not query or not query.strip():
        return []

    candidates: list[str] = []
    candidates.extend(extract_named_phrases(query))
    candidates.extend(_phrase_after_markers(query))
    candidates.extend(_alphanumeric_spans(query))

    ordered: list[str] = []
    seen: set[str] = set()
    for phrase in sorted(candidates, key=len, reverse=True):
        cleaned = " ".join(_raw_tokens(phrase)) if phrase else ""
        if not cleaned:
            continue
        folded = cleaned.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        ordered.append(cleaned)
    return ordered


@dataclass(frozen=True)
class QueryMatchProfile:
    """Lookup phrases and fallback tokens derived from a user query."""

    phrases: tuple[str, ...]
    tokens: tuple[str, ...]


def build_query_match_profile(query: str) -> QueryMatchProfile:
    phrases = extract_lookup_phrases(query)
    phrase_tokens: list[str] = []
    for phrase in phrases:
        for token in _TOKEN_RE.findall(phrase):
            if _is_distinctive_token(token) or _has_title_short_or_digit(token):
                phrase_tokens.append(token.casefold())

    long_tokens = [
        token.casefold()
        for token in _TOKEN_RE.findall(query or "")
        if len(token) >= 3 and _is_distinctive_token(token)
    ]

    merged_tokens: list[str] = []
    seen_tokens: set[str] = set()
    for token in phrase_tokens + long_tokens:
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        merged_tokens.append(token)

    return QueryMatchProfile(
        phrases=tuple(phrases),
        tokens=tuple(merged_tokens),
    )
