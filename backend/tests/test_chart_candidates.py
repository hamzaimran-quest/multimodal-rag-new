"""Tests for chart table candidate relevance ranking."""

from __future__ import annotations

from app.charts.candidates import rank_chart_table_candidates
from app.retrieval.models import RetrievedChunk


def _table_chunk(*, chunk_id: str, content: str, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="d1",
        filename="report.pdf",
        page_number=1,
        chunk_type="table",
        content=content,
        score=score,
    )


def test_rank_chart_table_candidates_prefers_query_matched_prior_chunk(monkeypatch):
    regional = _table_chunk(
        chunk_id="regional-t1",
        content="| Metric | 2024 | 2025 |\n| --- | --- | --- |\n| China | 100 | 110 |",
    )
    segment = _table_chunk(
        chunk_id="segment-t1",
        content="| Metric | 2024 | 2025 |\n| --- | --- | --- |\n| ICT Infrastructure | 100 | 110 |",
    )

    def fake_embed(texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "ict infrastructure" in lowered:
                vectors.append([1.0, 0.0])
            elif "china" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        return vectors

    monkeypatch.setattr("app.charts.candidates.embed_texts", fake_embed)

    ranked = rank_chart_table_candidates(
        [regional, segment],
        "ICT infrastructure revenue chart",
    )
    assert ranked[0].chunk_id == "segment-t1"
