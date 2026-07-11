"""Expand XLSX retrieval hits using query-resolved anchor entity keys."""

from __future__ import annotations

import logging
from typing import Any

from opensearchpy import OpenSearch

from app.config import settings
from app.ingestion.xlsx_entity_keys import resolve_anchor_keys_from_chunk
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


def fetch_chunks_for_anchor(
    client: OpenSearch,
    *,
    doc_id: str,
    anchor_key: str,
    user_id: int,
    sheet_names: list[str] | None = None,
    max_chunks: int | None = None,
) -> list[RetrievedChunk]:
    cap = max_chunks or settings.excel_cluster_expand_per_anchor
    filters: list[dict[str, Any]] = [
        {"term": {"user_id": str(user_id)}},
        {"term": {"doc_id": doc_id}},
        {"terms": {"extra_metadata.entity_keys": [anchor_key]}},
    ]
    if sheet_names:
        filters.append({"terms": {"extra_metadata.sheet_name": sheet_names}})

    body: dict[str, Any] = {
        "size": cap,
        "query": {"bool": {"filter": filters}},
    }
    response = client.search(index=settings.chunks_index, body=body)
    return [parse_search_hit(hit) for hit in response["hits"]["hits"]]


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


def _sheet_names_for_anchor(
    doc_id: str,
    *,
    workbook_schema: dict[str, Any] | None,
    chunk_sheet_map: dict[str, str],
) -> list[str] | None:
    names = set(_linked_sheets_from_schema(workbook_schema))
    names.update(chunk_sheet_map)
    if not names:
        return None
    return sorted(names)


def expand_xlsx_chunks_by_entity_keys(
    client: OpenSearch,
    chunks: list[RetrievedChunk],
    *,
    query: str,
    user_id: int,
    doc_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Merge anchor-linked workbook chunks into the retrieval set."""
    if not chunks:
        return chunks

    existing_ids = {chunk.chunk_id for chunk in chunks}
    merged = list(chunks)
    added = 0

    schema_cache: dict[str, dict[str, Any] | None] = {}
    chunk_sheet_maps = _linked_sheets_from_chunks(chunks)
    anchors = _resolve_anchor_targets(chunks, query)

    if anchors:
        for doc_id, anchor_key, _score in anchors:
            workbook_schema = _workbook_schema_for_doc(
                client,
                doc_id=doc_id,
                user_id=user_id,
                cache=schema_cache,
            )
            sheet_names = _sheet_names_for_anchor(
                doc_id,
                workbook_schema=workbook_schema,
                chunk_sheet_map=chunk_sheet_maps.get(doc_id, {}),
            )
            expanded = fetch_chunks_for_anchor(
                client,
                doc_id=doc_id,
                anchor_key=anchor_key,
                user_id=user_id,
                sheet_names=sheet_names,
            )
            for chunk in expanded:
                if chunk.chunk_id in existing_ids:
                    continue
                merged.append(chunk)
                existing_ids.add(chunk.chunk_id)
                added += 1
    else:
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

    if added:
        logger.info(
            "XLSX_ENTITY_EXPAND anchors=%s added=%s total=%s query_preview=%r",
            len(anchors),
            added,
            len(merged),
            (query or "")[:80],
        )
    return merged
