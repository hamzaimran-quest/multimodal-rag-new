"""Relevance ranking for chart table candidates."""

from __future__ import annotations

from app.ingestion.embeddings import embed_texts
from app.ingestion.xlsx_serialize import format_chunk_content_for_llm
from app.retrieval.models import RetrievedChunk

_SUMMARY_MAX_CHARS = 2000


def _token_set(text: str) -> set[str]:
    return {token.lower() for token in text.split() if token}


def _lexical_overlap_score(query: str, content: str) -> float:
    query_tokens = _token_set(query)
    if not query_tokens:
        return 0.0
    content_tokens = _token_set(content)
    return len(query_tokens & content_tokens) / len(query_tokens)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _candidate_summary(chunk: RetrievedChunk) -> str:
    content = format_chunk_content_for_llm(chunk.content, chunk.extra_metadata)
    if len(content) <= _SUMMARY_MAX_CHARS:
        return content
    return content[:_SUMMARY_MAX_CHARS]


def score_table_candidate_relevance(chunk: RetrievedChunk, query: str) -> float:
    """Higher scores indicate a better query-to-table match."""
    summary = _candidate_summary(chunk)
    lexical = _lexical_overlap_score(query, summary)
    retrieval = max(0.0, float(chunk.score))
    return lexical + retrieval * 0.05


def rank_chart_table_candidates(
    candidates: list[RetrievedChunk],
    query: str,
) -> list[RetrievedChunk]:
    """Order table chunks by query relevance before chart construction attempts."""
    query = query.strip()
    if len(candidates) <= 1 or not query:
        return candidates

    summaries = [_candidate_summary(chunk) for chunk in candidates]
    vectors = embed_texts([query, *summaries])
    query_vector = vectors[0]

    scored: list[tuple[float, int, RetrievedChunk]] = []
    for index, chunk in enumerate(candidates):
        similarity = _dot(query_vector, vectors[index + 1])
        lexical = _lexical_overlap_score(query, summaries[index])
        retrieval = max(0.0, float(chunk.score))
        score = similarity + lexical * 0.25 + retrieval * 0.05
        scored.append((score, index, chunk))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chunk for _, _, chunk in scored]
