"""Tests for the create_chart agent tool."""

from __future__ import annotations

import json

import pytest

from app.charts.candidates import chart_follow_up_on_priors, rank_chart_table_candidates
from app.llm.tools import execute_create_chart
from app.retrieval.models import RetrievedChunk


def _table_chunk(*, chunk_id: str = "t1", content: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="d1",
        filename="report.pdf",
        page_number=4,
        chunk_type="table",
        content=content,
        score=score,
    )


def test_create_chart_returns_bar_chart(monkeypatch):
    content = (
        "| Metric | 2022 | 2023 | 2024 |\n"
        "| --- | --- | --- | --- |\n"
        "| Series A | 100 | 110 | 120 |\n"
        "| Series B | 10 | 12 | 15 |"
    )
    chunk = _table_chunk(content=content)

    def fake_retrieve(client, query, user_id, top_k=None, doc_id=None, doc_ids=None):
        from app.retrieval.models import SearchResponse

        effective = doc_ids or ([doc_id] if doc_id else None)
        return SearchResponse(
            query=query,
            top_k=top_k or 8,
            doc_id=effective[0] if effective and len(effective) == 1 else None,
            doc_ids=effective,
            total=1,
            results=[chunk],
        )

    llm_spec = {
        "chart_type": "bar",
        "title": "Series A",
        "labels": ["2022", "2023", "2024"],
        "series": [{"name": "Series A", "values": [100.0, 110.0, 120.0]}],
    }

    monkeypatch.setattr("app.llm.tools.hybrid_retrieve", fake_retrieve)
    monkeypatch.setattr(
        "app.charts.build.extract_chart_data_spec",
        lambda user_query, markdown, chart_type=None: (llm_spec, None),
    )
    monkeypatch.setattr(
        "app.charts.build.build_quickchart_url",
        lambda config: "https://quickchart.io/chart?c=series",
    )

    payload_json, charts, source_chunks = execute_create_chart(
        object(),
        user_id=1,
        query="series revenue",
        chart_type="bar",
    )
    payload = json.loads(payload_json)
    assert payload["status"] == "created"
    assert len(charts) == 1
    assert len(source_chunks) == 1
    assert charts[0]["chart_type"] == "bar"
    assert charts[0]["chart_url"] == "https://quickchart.io/chart?c=series"
    assert charts[0]["derivation"] == "tool"


def test_create_chart_rejects_pie(monkeypatch):
    payload_json, charts, _ = execute_create_chart(
        object(),
        user_id=1,
        query="region revenue",
        chart_type="pie",
    )
    payload = json.loads(payload_json)
    assert payload["error"] == "invalid_chart_type"
    assert charts == []


def test_create_chart_reports_not_chartable(monkeypatch):
    text_chunk = RetrievedChunk(
        chunk_id="x1",
        doc_id="d1",
        filename="report.pdf",
        page_number=1,
        chunk_type="text",
        content="plain text",
        score=0.5,
    )

    def fake_retrieve(client, query, user_id, top_k=None, doc_id=None, doc_ids=None):
        from app.retrieval.models import SearchResponse

        return SearchResponse(query=query, top_k=top_k or 8, doc_id=doc_id, doc_ids=doc_ids, total=1, results=[text_chunk])

    monkeypatch.setattr("app.llm.tools.hybrid_retrieve", fake_retrieve)

    payload_json, charts, _ = execute_create_chart(
        object(),
        user_id=1,
        query="anything",
        chart_type="bar",
    )
    payload = json.loads(payload_json)
    assert payload["status"] == "not_chartable"
    assert "cannot be created" in payload["message"].lower()
    assert charts == []


def test_create_chart_uses_prior_table_chunk_when_search_misses(monkeypatch):
    regional = _table_chunk(
        chunk_id="regional-t1",
        content="| Region | 2024 | 2025 |\n| --- | --- | --- |\n| China | 100 | 110 |",
    )
    wrong = _table_chunk(
        chunk_id="wrong-t1",
        content=(
            "| Metric | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            "| Revenue | 1 | 2 | 3 | 4 | 5 | 6 | 7 |"
        ),
    )

    def fake_retrieve(client, query, user_id, top_k=None, doc_id=None, doc_ids=None):
        from app.retrieval.models import SearchResponse

        return SearchResponse(query=query, top_k=top_k or 8, total=1, results=[wrong])

    def fake_get_chunk(client, chunk_id, user_id):
        if chunk_id == "regional-t1":
            return regional
        return None

    def fake_attempt(chunk, *, user_query, chart_type=None):
        if chunk.chunk_id != "regional-t1":
            return None, "Table is not chartable."
        return (
            {
                "chart_type": "bar",
                "chart_url": "https://quickchart.io/chart?c=regional",
                "chunk_id": chunk.chunk_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "doc_id": chunk.doc_id,
                "derivation": "tool",
                "citation": {
                    "chunk_id": chunk.chunk_id,
                    "filename": chunk.filename,
                    "page_number": chunk.page_number,
                    "chunk_type": chunk.chunk_type,
                },
            },
            None,
        )

    monkeypatch.setattr("app.llm.tools.hybrid_retrieve", fake_retrieve)
    monkeypatch.setattr("app.llm.tools.get_chunk_for_user", fake_get_chunk)
    monkeypatch.setattr("app.llm.tools.attempt_chart_from_chunk", fake_attempt)

    payload_json, charts, source_chunks = execute_create_chart(
        object(),
        user_id=1,
        query="regional revenue",
        chart_type="bar",
        prior_table_chunk_ids=["regional-t1"],
    )
    payload = json.loads(payload_json)
    assert payload["status"] == "created"
    assert charts[0]["chunk_id"] == "regional-t1"
    assert source_chunks[0].chunk_id == "regional-t1"


def test_create_chart_tries_multiple_search_candidates(monkeypatch):
    bad = _table_chunk(
        chunk_id="bad-t1",
        content="| Metric | 2021 | 2022 | 2023 | 2024 | 2025 |\n| --- | --- | --- | --- | --- | --- |\n| Revenue | 1 | 2 | 3 | 4 | 5 |",
    )
    good = _table_chunk(
        chunk_id="good-t1",
        content="| Segment | 2024 | 2025 |\n| --- | --- | --- |\n| ICT | 100 | 110 |",
    )

    def fake_retrieve(client, query, user_id, top_k=None, doc_id=None, doc_ids=None):
        from app.retrieval.models import SearchResponse

        return SearchResponse(query=query, top_k=top_k or 8, total=2, results=[bad, good])

    call_count = {"n": 0}

    def fake_attempt(chunk, *, user_query, chart_type=None):
        call_count["n"] += 1
        if chunk.chunk_id == "bad-t1":
            return None, "Series 0 length does not match labels."
        return (
            {
                "chart_type": "bar",
                "chart_url": "https://quickchart.io/chart?c=good",
                "chunk_id": chunk.chunk_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "doc_id": chunk.doc_id,
                "derivation": "tool",
                "citation": {
                    "chunk_id": chunk.chunk_id,
                    "filename": chunk.filename,
                    "page_number": chunk.page_number,
                    "chunk_type": chunk.chunk_type,
                },
            },
            None,
        )

    monkeypatch.setattr("app.llm.tools.hybrid_retrieve", fake_retrieve)
    monkeypatch.setattr("app.llm.tools.attempt_chart_from_chunk", fake_attempt)

    payload_json, charts, _ = execute_create_chart(
        object(),
        user_id=1,
        query="segment revenue",
        chart_type="bar",
    )
    payload = json.loads(payload_json)
    assert payload["status"] == "created"
    assert charts[0]["chunk_id"] == "good-t1"
    assert call_count["n"] == 2


def test_create_chart_ranks_prior_chunks_by_query(monkeypatch):
    regional = _table_chunk(
        chunk_id="regional-t1",
        content="| Metric | 2024 | 2025 |\n| --- | --- | --- |\n| China | 100 | 110 |",
        score=0.0,
    )
    segment = _table_chunk(
        chunk_id="segment-t1",
        content="| Metric | 2024 | 2025 |\n| --- | --- | --- |\n| ICT Infrastructure | 100 | 110 |",
        score=0.0,
    )

    def fake_get_chunk(client, chunk_id, user_id):
        return {"regional-t1": regional, "segment-t1": segment}.get(chunk_id)

    def fake_retrieve(client, query, user_id, top_k=None, doc_id=None, doc_ids=None):
        from app.retrieval.models import SearchResponse

        return SearchResponse(query=query, top_k=top_k or 8, total=0, results=[])

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

    monkeypatch.setattr("app.llm.tools.get_chunk_for_user", fake_get_chunk)
    monkeypatch.setattr("app.llm.tools.hybrid_retrieve", fake_retrieve)
    monkeypatch.setattr("app.charts.candidates.embed_texts", fake_embed)
    monkeypatch.setattr(
        "app.charts.build.build_quickchart_url",
        lambda config: "https://quickchart.io/chart?c=segment",
    )

    payload_json, charts, _ = execute_create_chart(
        object(),
        user_id=1,
        query="ICT infrastructure revenue chart",
        chart_type="bar",
        prior_table_chunk_ids=["regional-t1", "segment-t1"],
    )
    payload = json.loads(payload_json)
    assert payload["status"] == "created"
    assert charts[0]["chunk_id"] == "segment-t1"


def test_chart_follow_up_on_priors_detects_metric_only_query():
    fsi = _table_chunk(
        chunk_id="fsi-t1",
        content="Somalia | 2023 | 5 | 109.8 | 9.8 | 9.5",
        score=0.0,
    )
    fsi.filename = "FSI-2023-DOWNLOAD.xlsx"
    assert chart_follow_up_on_priors("plot the first 5 metrics", [fsi]) is True
    assert chart_follow_up_on_priors("Somalia demographic pressures chart", [fsi]) is False


def test_create_chart_follow_up_uses_priors_without_search(monkeypatch):
    fsi = _table_chunk(
        chunk_id="fsi-t1",
        content="Somalia | 2023 | 5 | 109.8 | 9.8 | 9.5 | 8.6 | 9.1 | 7.5",
        score=0.0,
    )
    fsi.filename = "FSI-2023-DOWNLOAD.xlsx"
    fsi.extra_metadata = {
        "source_format": "xlsx",
        "content_format": "slim_rows",
        "table_headers": [
            "Country",
            "Year",
            "Rank",
            "Total",
            "S1: Demographic Pressures",
            "S2: Refugees and IDPs",
            "C3: Group Grievance",
            "E3: Human Flight and Brain Drain",
            "E2: Economic Inequality",
        ],
        "entity_key_column": "Country",
        "entity_keys": ["Somalia"],
    }
    netflix = _table_chunk(
        chunk_id="netflix-t1",
        content="| duration_minutes | release_year |\n| --- | --- |\n| 100 | 2019 |",
        score=0.9,
    )
    netflix.filename = "netflix_titles.xlsx"
    netflix.extra_metadata = {"source_format": "xlsx"}

    def fake_get_chunk(client, chunk_id, user_id):
        return {"fsi-t1": fsi}.get(chunk_id)

    def fail_retrieve(*args, **kwargs):
        raise AssertionError("hybrid_retrieve must not run for chart follow-ups on priors")

    monkeypatch.setattr("app.llm.tools.get_chunk_for_user", fake_get_chunk)
    monkeypatch.setattr("app.llm.tools.hybrid_retrieve", fail_retrieve)
    monkeypatch.setattr(
        "app.charts.build.build_quickchart_url",
        lambda config: "https://quickchart.io/chart?c=fsi",
    )

    payload_json, charts, source_chunks = execute_create_chart(
        object(),
        user_id=1,
        query="plot the first 5 metrics",
        prior_table_chunk_ids=["fsi-t1"],
    )
    payload = json.loads(payload_json)
    assert payload["status"] == "created"
    assert charts[0]["chunk_id"] == "fsi-t1"
    assert source_chunks[0].chunk_id == "fsi-t1"


def test_rank_chart_table_candidates_boosts_prior_chunks(monkeypatch):
    prior = _table_chunk(
        chunk_id="prior-t1",
        content="| Metric | 2024 | 2025 |\n| --- | --- | --- |\n| ICT | 100 | 110 |",
        score=0.0,
    )
    other = _table_chunk(
        chunk_id="other-t1",
        content="| Metric | 2024 | 2025 |\n| --- | --- | --- |\n| China | 200 | 220 |",
        score=0.9,
    )

    def fake_embed(texts):
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("app.charts.candidates.embed_texts", fake_embed)
    ranked = rank_chart_table_candidates(
        [other, prior],
        "segment revenue chart",
        prior_chunk_ids={"prior-t1"},
    )
    assert ranked[0].chunk_id == "prior-t1"


@pytest.mark.asyncio
async def test_agent_create_chart_tool(monkeypatch) -> None:
    from app.llm.agent import iter_agent_turn
    from tests.test_agent import _router_then_stop

    chart_round = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_chart",
                            "type": "function",
                            "function": {
                                "name": "create_chart",
                                "arguments": json.dumps(
                                    {"query": "segment revenue", "chart_type": "bar", "chunk_id": "t1"}
                                ),
                            },
                        }
                    ],
                }
            }
        ]
    }

    def fake_create_chart(
        client,
        user_id,
        query,
        chart_type=None,
        scope_doc_ids=None,
        chunk_id=None,
        prior_table_chunk_ids=None,
        top_k=None,
    ):
        chart = {
            "chart_type": "bar",
            "chunk_id": "t1",
            "filename": "report.pdf",
            "page_number": 4,
            "periods": ["2024"],
            "series": [{"name": "A", "values": [1.0]}],
            "derivation": "tool",
        }
        return json.dumps({"status": "created", "chart_type": "bar"}), [chart], []

    monkeypatch.setattr("app.llm.agent.groq_chat_completion", _router_then_stop(chart_round))
    monkeypatch.setattr("app.llm.agent.execute_create_chart", fake_create_chart)

    turn = None
    async for event in iter_agent_turn(
        client=object(),
        user_id=1,
        user_query="create a bar chart of segment revenue",
    ):
        if event["type"] == "complete":
            turn = event["result"]

    assert turn is not None
    assert "create_chart" in turn.tools_used
    assert len(turn.tool_charts) == 1
    assert turn.tool_charts[0]["chart_type"] == "bar"
