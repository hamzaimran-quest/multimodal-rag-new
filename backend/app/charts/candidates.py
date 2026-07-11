"""Relevance ranking for chart table candidates."""

from __future__ import annotations

from app.charts.excel_build import excel_entity_match_score
from app.ingestion.embeddings import embed_texts
from app.ingestion.xlsx_serialize import format_chunk_content_for_llm
from app.retrieval.models import RetrievedChunk

_SUMMARY_MAX_CHARS = 2000
_PRIOR_CHUNK_BOOST = 1.0
_FOLLOW_UP_OVERLAP_THRESHOLD = 0.2


def _token_set(text: str) -> set[str]:
    return {token.lower() for token in text.split() if token}


def _semantic_query_tokens(query: str) -> set[str]:
    return {token for token in _token_set(query) if not token.isdigit()}


def _semantic_lexical_overlap_score(query: str, content: str) -> float:
    query_tokens = _semantic_query_tokens(query)
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


def query_lexical_overlap_with_chunks(query: str, chunks: list[RetrievedChunk]) -> float:
    """Max semantic token overlap between query and any chunk summary."""
    if not chunks:
        return 0.0
    return max(_semantic_lexical_overlap_score(query, _candidate_summary(chunk)) for chunk in chunks)


def _lexical_overlap_score(query: str, content: str) -> float:
    query_tokens = _token_set(query)
    if not query_tokens:
        return 0.0
    content_tokens = _token_set(content)
    return len(query_tokens & content_tokens) / len(query_tokens)


def chart_follow_up_on_priors(query: str, prior_chunks: list[RetrievedChunk]) -> bool:
    """
    True when the user is refining a chart on tables already in session context
    (e.g. "plot the first 5 metrics") rather than searching for a new table.
    """
    from app.charts.auto import chart_requested

    if not prior_chunks or not chart_requested(query):
        return False
    return query_lexical_overlap_with_chunks(query, prior_chunks) < _FOLLOW_UP_OVERLAP_THRESHOLD


def score_table_candidate_relevance(chunk: RetrievedChunk, query: str) -> float:
    """Higher scores indicate a better query-to-table match."""
    summary = _candidate_summary(chunk)
    lexical = _lexical_overlap_score(query, summary)
    retrieval = max(0.0, float(chunk.score))
    excel_bonus = excel_entity_match_score(chunk, query)
    return lexical + retrieval * 0.05 + excel_bonus


def rank_chart_table_candidates(
    candidates: list[RetrievedChunk],
    query: str,
    *,
    prior_chunk_ids: set[str] | None = None,
) -> list[RetrievedChunk]:
    """Order table chunks by query relevance before chart construction attempts."""
    query = query.strip()
    if len(candidates) <= 1:
        return candidates

    prior_ids = prior_chunk_ids or set()
    summaries = [_candidate_summary(chunk) for chunk in candidates]

    if query:
        vectors = embed_texts([query, *summaries])
        query_vector = vectors[0]
    else:
        vectors = None
        query_vector = None

    scored: list[tuple[float, int, RetrievedChunk]] = []
    for index, chunk in enumerate(candidates):
        similarity = _dot(query_vector, vectors[index + 1]) if query_vector is not None else 0.0
        lexical = _lexical_overlap_score(query, summaries[index]) if query else 0.0
        retrieval = max(0.0, float(chunk.score))
        prior_boost = _PRIOR_CHUNK_BOOST if chunk.chunk_id in prior_ids else 0.0
        excel_bonus = excel_entity_match_score(chunk, query)
        score = similarity + lexical * 0.25 + retrieval * 0.05 + prior_boost + excel_bonus
        scored.append((score, index, chunk))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chunk for _, _, chunk in scored]
