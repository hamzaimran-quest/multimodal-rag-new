"""Preserve named entities in retrieval queries (titles, labels with colons, etc.)."""

from __future__ import annotations

import re

_QUOTED_PHRASE_RE = re.compile(r'"([^"]{2,120})"|\'([^\']{2,120})\'')
_COLON_PHRASE_RE = re.compile(r"([^\n,;]{2,80}:\s*[^\n,;]+)")
_TITLE_CASE_RUN_RE = re.compile(
    r"\b(?:[A-Z][a-z0-9]+(?:['’][a-z]+)?(?:\s+(?:[A-Z][a-z0-9]+|&|of|the|in|on|at|for|and|a|an))+)\b"
)


def extract_named_phrases(text: str) -> list[str]:
    """Pull distinctive multi-word phrases likely to identify a row or entity."""
    if not text or not text.strip():
        return []

    candidates: list[str] = []
    for match in _QUOTED_PHRASE_RE.finditer(text):
        phrase = (match.group(1) or match.group(2) or "").strip()
        if phrase:
            candidates.append(phrase)

    for match in _COLON_PHRASE_RE.finditer(text):
        phrase = match.group(1).strip()
        if phrase:
            candidates.append(phrase)

    for match in _TITLE_CASE_RUN_RE.finditer(text):
        phrase = match.group(0).strip()
        if len(phrase.split()) >= 2:
            candidates.append(phrase)

    ordered: list[str] = []
    seen: set[str] = set()
    for phrase in sorted(candidates, key=len, reverse=True):
        folded = phrase.casefold()
        if folded in seen or len(phrase) < 4:
            continue
        seen.add(folded)
        ordered.append(phrase)
    return ordered


def merge_retrieval_anchor_phrases(
    query: str,
    *,
    fallback_queries: list[str] | None = None,
) -> str:
    """Ensure named phrases from fallback text survive in the retrieval query."""
    target = (query or "").strip()
    if not target:
        return query

    sources = [target]
    for fallback in fallback_queries or []:
        cleaned = (fallback or "").strip()
        if cleaned:
            sources.append(cleaned)

    target_fold = target.casefold()
    missing: list[str] = []
    seen: set[str] = set()
    for source in sources:
        for phrase in extract_named_phrases(source):
            folded = phrase.casefold()
            if folded in target_fold or folded in seen:
                continue
            seen.add(folded)
            missing.append(phrase)

    if not missing:
        return target

    merged = " ".join(missing + [target]).strip()
    return merged
