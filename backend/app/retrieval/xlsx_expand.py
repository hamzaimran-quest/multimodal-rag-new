"""Expand XLSX retrieval hits using query-resolved anchor entity keys."""

from __future__ import annotations

import logging
from typing import Any

from opensearchpy import OpenSearch

from app.config import settings
from app.ingestion.xlsx_entity_keys import resolve_anchor_keys_from_chunk
from app.retrieval.query_phrases import build_query_match_profile
from app.opensearch.documents import get_document_for_user
from app.retrieval.models import RetrievedChunk
from app.retrieval.service import parse_search_hit

logger = logging.getLogger(__name__)


def _collect_legacy_entity_keys(chunks: list[RetrievedChunk]) -> set[str]:
    keys: set[str] = set()
    for chunk in chunks:
        extra = chunk.extra_metadata or {}
        if extra.get("source_format") != "xlsx":
            continue
        single = extra.get("entity_key")
        if single:
            keys.add(str(single))
        for value in extra.get("entity_keys") or []:
            if value:
                keys.add(str(value))
    return keys


def _linked_sheets_from_schema(workbook_schema: dict[str, Any] | None) -> dict[str, str]:
    if not workbook_schema:
        return {}

    sheets: dict[str, str] = {}
    for cluster in workbook_schema.get("clusters") or []:
        primary_sheet = cluster.get("primary_sheet")
        primary_key = cluster.get("primary_key_column")
        if primary_sheet and primary_key:
            sheets[str(primary_sheet)] = str(primary_key)
        for satellite in cluster.get("satellites") or []:
            sheet_name = satellite.get("sheet")
            key_column = satellite.get("key_column")
            if sheet_name and key_column:
                sheets[str(sheet_name)] = str(key_column)
    return sheets


def _workbook_sheet_names(workbook_schema: dict[str, Any] | None) -> set[str]:
    if not workbook_schema:
        return set()
    names = {str(name) for name in workbook_schema.get("standalone_sheets") or [] if name}
    for cluster in workbook_schema.get("clusters") or []:
        primary_sheet = cluster.get("primary_sheet")
        if primary_sheet:
            names.add(str(primary_sheet))
        for satellite in cluster.get("satellites") or []:
            sheet_name = satellite.get("sheet")
            if sheet_name:
                names.add(str(sheet_name))
    return names


def _distinct_xlsx_sheet_names(chunks: list[RetrievedChunk]) -> dict[str, set[str]]:
    by_doc: dict[str, set[str]] = {}
    for chunk in chunks:
        extra = chunk.extra_metadata or {}
        if extra.get("source_format") != "xlsx":
            continue
        sheet_name = extra.get("sheet_name")
        if not sheet_name:
            continue
        by_doc.setdefault(chunk.doc_id, set()).add(str(sheet_name))
    return by_doc


def _is_multisheet_workbook(
    client: OpenSearch,
    *,
    doc_id: str,
    user_id: int,
    chunks: list[RetrievedChunk],
    schema_cache: dict[str, dict[str, Any] | None],
) -> bool:
    schema = _workbook_schema_for_doc(
        client,
        doc_id=doc_id,
        user_id=user_id,
        cache=schema_cache,
    )
    sheet_names = _workbook_sheet_names(schema)
    if len(sheet_names) > 1:
        return True
    return len(_distinct_xlsx_sheet_names(chunks).get(doc_id, set())) > 1


def _should_skip_legacy_expand(
    client: OpenSearch,
    chunks: list[RetrievedChunk],
    *,
    user_id: int,
    doc_ids: list[str] | None,
    query: str,
    anchor_fallback_query: str | None,
) -> bool:
    """Skip bulk legacy expansion when a multi-sheet workbook needs anchor resolution."""
    anchor_text = " ".join(
        part for part in [anchor_fallback_query, query] if part and part.strip()
    ).strip()
    profile = build_query_match_profile(anchor_text)
    if not profile.phrases and len(profile.tokens) < 2:
        return False

    schema_cache: dict[str, dict[str, Any] | None] = {}
    xlsx_doc_ids = {
        chunk.doc_id
        for chunk in chunks
        if (chunk.extra_metadata or {}).get("source_format") == "xlsx"
    }
    if doc_ids:
        xlsx_doc_ids.update(doc_ids)

    for doc_id in xlsx_doc_ids:
        if _is_multisheet_workbook(
            client,
            doc_id=doc_id,
            user_id=user_id,
            chunks=chunks,
            schema_cache=schema_cache,
        ):
            return True
    return False


def _tag_anchor_expanded_chunk(chunk: RetrievedChunk, *, anchor_key: str) -> RetrievedChunk:
    extra = dict(chunk.extra_metadata or {})
    extra["xlsx_anchor_expanded"] = True
    extra["xlsx_anchor_key"] = anchor_key
    return chunk.model_copy(update={"extra_metadata": extra, "score": 0.0})


def _soft_links_from_schema(workbook_schema: dict[str, Any] | None) -> dict[str, str]:
    if not workbook_schema:
        return {}
    sheets: dict[str, str] = {}
    for link in workbook_schema.get("soft_links") or []:
        sheet_name = link.get("sheet")
        key_column = link.get("key_column")
        if sheet_name and key_column:
            sheets[str(sheet_name)] = str(key_column)
    return sheets


def _standalone_fk_links_from_schema(workbook_schema: dict[str, Any] | None) -> dict[str, str]:
    """Standalone sheets that share a cluster FK column (persisted or inferred)."""
    if not workbook_schema:
        return {}

    sheets: dict[str, str] = {}
    for link in workbook_schema.get("standalone_fk_links") or []:
        sheet_name = link.get("sheet")
        key_column = link.get("key_column")
        if sheet_name and key_column:
            sheets[str(sheet_name)] = str(key_column)

    if sheets:
        return sheets

    # Back-compat for documents indexed before standalone_fk_links existed.
    cluster_keys: dict[str, str] = {}
    for cluster in workbook_schema.get("clusters") or []:
        primary_key = cluster.get("primary_key_column")
        if primary_key:
            cluster_keys[str(cluster.get("primary_sheet") or "")] = str(primary_key)

    primary_key_columns = {key for key in cluster_keys.values() if key}
    if not primary_key_columns:
        return sheets

    for sheet_name in workbook_schema.get("standalone_sheets") or []:
        # Without header metadata we only know the sheet name; assume the cluster FK
        # column name when there is a single primary key across clusters.
        if len(primary_key_columns) == 1:
            sheets[str(sheet_name)] = next(iter(primary_key_columns))
    return sheets


def linked_sheets_for_anchor(
    workbook_schema: dict[str, Any] | None,
    chunk_sheet_map: dict[str, str],
) -> dict[str, str]:
    """All sheets that may hold FK-linked rows for an anchor entity."""
    sheets = _linked_sheets_from_schema(workbook_schema)
    sheets.update(_soft_links_from_schema(workbook_schema))
    sheets.update(_standalone_fk_links_from_schema(workbook_schema))
    sheets.update(chunk_sheet_map)
    return sheets


def _linked_sheets_from_chunks(chunks: list[RetrievedChunk]) -> dict[str, dict[str, str]]:
    """Per doc_id: sheet_name -> entity_key_column discovered from chunk metadata."""
    by_doc: dict[str, dict[str, str]] = {}
    for chunk in chunks:
        extra = chunk.extra_metadata or {}
        if extra.get("source_format") != "xlsx":
            continue
        sheet_name = extra.get("sheet_name")
        key_column = extra.get("entity_key_column")
        if not sheet_name or not key_column:
            continue
        by_doc.setdefault(chunk.doc_id, {})[str(sheet_name)] = str(key_column)
    return by_doc


def _resolve_anchor_targets(
    chunks: list[RetrievedChunk],
    query: str,
) -> list[tuple[str, str, float]]:
    """Return [(doc_id, entity_key, score)] sorted by descending score."""
    best: dict[tuple[str, str], float] = {}
    for chunk in chunks:
        for entity_key, score in resolve_anchor_keys_from_chunk(chunk, query):
            compound = (chunk.doc_id, entity_key)
            best[compound] = max(best.get(compound, 0.0), score * max(chunk.score, 0.01))

    ranked = sorted(
        ((doc_id, entity_key, score) for (doc_id, entity_key), score in best.items()),
        key=lambda item: item[2],
        reverse=True,
    )
    cap = max(1, settings.excel_anchor_max_entities)
    return ranked[:cap]


def fetch_chunks_by_entity_keys(
    client: OpenSearch,
    *,
    entity_keys: set[str],
    user_id: int,
    doc_ids: list[str] | None = None,
    max_chunks: int | None = None,
) -> list[RetrievedChunk]:
    if not entity_keys:
        return []

    cap = max_chunks or settings.excel_cluster_expand_per_anchor
    filters: list[dict[str, Any]] = [{"term": {"user_id": str(user_id)}}]
    if doc_ids:
        if len(doc_ids) == 1:
            filters.append({"term": {"doc_id": doc_ids[0]}})
        else:
            filters.append({"terms": {"doc_id": doc_ids}})

    key_list = sorted(entity_keys)
    body: dict[str, Any] = {
        "size": cap,
        "query": {
            "bool": {
                "filter": filters,
                "should": [
                    {"terms": {"extra_metadata.entity_keys": key_list}},
                    {"terms": {"extra_metadata.entity_key": key_list}},
                ],
                "minimum_should_match": 1,
            }
        },
    }
    response = client.search(index=settings.chunks_index, body=body)
    return [parse_search_hit(hit) for hit in response["hits"]["hits"]]


def fetch_chunk_for_anchor_on_sheet(
    client: OpenSearch,
    *,
    doc_id: str,
    anchor_key: str,
    sheet_name: str,
    user_id: int,
) -> RetrievedChunk | None:
    body: dict[str, Any] = {
        "size": 1,
        "query": {
            "bool": {
                "filter": [
                    {"term": {"user_id": str(user_id)}},
                    {"term": {"doc_id": doc_id}},
                    {"term": {"extra_metadata.sheet_name": sheet_name}},
                    {"terms": {"extra_metadata.entity_keys": [anchor_key]}},
                ]
            }
        },
    }
    response = client.search(index=settings.chunks_index, body=body)
    hits = response["hits"]["hits"]
    if not hits:
        return None
    return parse_search_hit(hits[0])


def _chunk_covers_anchor(chunk: RetrievedChunk, anchor_key: str) -> bool:
    extra = chunk.extra_metadata or {}
    entity_keys = {str(value) for value in (extra.get("entity_keys") or []) if value}
    if anchor_key in entity_keys:
        return True
    row_entity_keys = extra.get("row_entity_keys") or {}
    return anchor_key in {str(value) for value in row_entity_keys.values() if value}


def _sheets_already_covering_anchor(
    chunks: list[RetrievedChunk],
    *,
    doc_id: str,
    anchor_key: str,
) -> set[str]:
    covered: set[str] = set()
    for chunk in chunks:
        if chunk.doc_id != doc_id:
            continue
        sheet_name = (chunk.extra_metadata or {}).get("sheet_name")
        if sheet_name and _chunk_covers_anchor(chunk, anchor_key):
            covered.add(str(sheet_name))
    return covered


def fetch_chunks_for_anchor_per_sheet(
    client: OpenSearch,
    *,
    doc_id: str,
    anchor_key: str,
    user_id: int,
    linked_sheets: dict[str, str],
    existing_chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Fetch at most one chunk per linked sheet that is not already represented."""
    already_covered = _sheets_already_covering_anchor(
        existing_chunks,
        doc_id=doc_id,
        anchor_key=anchor_key,
    )
    expanded: list[RetrievedChunk] = []
    for sheet_name in sorted(linked_sheets):
        if sheet_name in already_covered:
            continue
        chunk = fetch_chunk_for_anchor_on_sheet(
            client,
            doc_id=doc_id,
            anchor_key=anchor_key,
            sheet_name=sheet_name,
            user_id=user_id,
        )
        if chunk is not None:
            expanded.append(chunk)
            already_covered.add(sheet_name)
    return expanded


def _workbook_schema_for_doc(
    client: OpenSearch,
    *,
    doc_id: str,
    user_id: int,
    cache: dict[str, dict[str, Any] | None],
) -> dict[str, Any] | None:
    if doc_id not in cache:
        record = get_document_for_user(client, doc_id, user_id)
        schema = (record or {}).get("workbook_schema")
        cache[doc_id] = schema if isinstance(schema, dict) else None
    return cache[doc_id]


def expand_xlsx_chunks_by_entity_keys(
    client: OpenSearch,
    chunks: list[RetrievedChunk],
    *,
    query: str,
    user_id: int,
    doc_ids: list[str] | None = None,
    anchor_fallback_query: str | None = None,
) -> tuple[list[RetrievedChunk], set[str]]:
    """Merge anchor-linked workbook chunks into the retrieval set."""
    if not chunks:
        return chunks, set()

    existing_ids = {chunk.chunk_id for chunk in chunks}
    merged = list(chunks)
    added = 0
    anchor_keys: set[str] = set()

    schema_cache: dict[str, dict[str, Any] | None] = {}
    chunk_sheet_maps = _linked_sheets_from_chunks(chunks)
    anchors = _resolve_anchor_targets(chunks, query)

    fallback = (anchor_fallback_query or "").strip()
    if not anchors and fallback and fallback.casefold() != (query or "").strip().casefold():
        anchors = _resolve_anchor_targets(chunks, fallback)
        if anchors:
            logger.info(
                "XLSX_ENTITY_EXPAND anchor_retry query_preview=%r fallback_preview=%r anchors=%s",
                (query or "")[:80],
                fallback[:80],
                len(anchors),
            )

    if anchors:
        for doc_id, anchor_key, _score in anchors:
            anchor_keys.add(anchor_key)
            workbook_schema = _workbook_schema_for_doc(
                client,
                doc_id=doc_id,
                user_id=user_id,
                cache=schema_cache,
            )
            linked_sheets = linked_sheets_for_anchor(
                workbook_schema,
                chunk_sheet_maps.get(doc_id, {}),
            )
            expanded = fetch_chunks_for_anchor_per_sheet(
                client,
                doc_id=doc_id,
                anchor_key=anchor_key,
                user_id=user_id,
                linked_sheets=linked_sheets,
                existing_chunks=merged,
            )
            for chunk in expanded:
                if chunk.chunk_id in existing_ids:
                    continue
                merged.append(_tag_anchor_expanded_chunk(chunk, anchor_key=anchor_key))
                existing_ids.add(chunk.chunk_id)
                added += 1
    elif not _should_skip_legacy_expand(
        client,
        merged,
        user_id=user_id,
        doc_ids=doc_ids,
        query=query,
        anchor_fallback_query=anchor_fallback_query,
    ):
        legacy_keys = _collect_legacy_entity_keys(chunks)
        if legacy_keys:
            expanded = fetch_chunks_by_entity_keys(
                client,
                entity_keys=legacy_keys,
                user_id=user_id,
                doc_ids=doc_ids,
                max_chunks=settings.excel_cluster_expand_per_anchor,
            )
            for chunk in expanded:
                if chunk.chunk_id in existing_ids:
                    continue
                merged.append(chunk)
                existing_ids.add(chunk.chunk_id)
                added += 1
    elif (query or "").strip() or fallback:
        logger.info(
            "XLSX_ENTITY_EXPAND legacy_skipped anchors=0 query_preview=%r fallback_preview=%r",
            (query or "")[:80],
            fallback[:80],
        )

    if added:
        logger.info(
            "XLSX_ENTITY_EXPAND anchors=%s added=%s total=%s query_preview=%r",
            len(anchor_keys),
            added,
            len(merged),
            (query or "")[:80],
        )
    return merged, anchor_keys


def chunk_covers_anchor(chunk: RetrievedChunk, anchor_key: str) -> bool:
    return _chunk_covers_anchor(chunk, anchor_key)
