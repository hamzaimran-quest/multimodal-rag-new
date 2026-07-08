"""OpenSearch index mappings and bootstrap helpers."""

import logging
import shutil
from typing import Any

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError

from app.config import settings

logger = logging.getLogger(__name__)


def _mapping_has_field(client: OpenSearch, index_name: str, field: str) -> bool:
    try:
        mapping = client.indices.get_mapping(index=index_name)
        properties = mapping[index_name]["mappings"].get("properties", {})
        return field in properties
    except NotFoundError:
        return False


def _wipe_local_document_storage() -> None:
    """Remove legacy unscoped upload/image directories after schema migration."""
    for root in (settings.resolved_uploads_dir, settings.resolved_images_dir):
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
    logger.warning("Wiped local uploads/images after OpenSearch schema migration (user_id required)")


def chunks_index_body() -> dict[str, Any]:
    dim = settings.embedding_dimension
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "user_id": {"type": "keyword"},
                "filename": {"type": "keyword"},
                "page_number": {"type": "integer"},
                "chunk_type": {"type": "keyword"},
                "content": {"type": "text", "analyzer": "standard"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "lucene",
                        "parameters": {"ef_construction": 128, "m": 16},
                    },
                },
                "bbox": {"type": "float"},
                "image_path": {"type": "keyword"},
                "upload_timestamp": {"type": "date"},
                "extra_metadata": {
                    "type": "object",
                    "enabled": True,
                },
            }
        },
    }


def documents_index_body() -> dict[str, Any]:
    return {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "user_id": {"type": "keyword"},
                "filename": {"type": "keyword"},
                "ingestion_status": {"type": "keyword"},
                "ingestion_progress": {"type": "float"},
                "progress_message": {"type": "text"},
                "upload_timestamp": {"type": "date"},
                "chunk_count": {"type": "integer"},
                "page_count": {"type": "integer"},
                "error_message": {"type": "text"},
            }
        },
    }


def ensure_indices(client: OpenSearch) -> dict[str, bool]:
    """Create chunk and document-registry indices if missing. Returns {index: created}."""
    for index_name in (settings.chunks_index, settings.documents_index):
        if client.indices.exists(index=index_name) and not _mapping_has_field(
            client, index_name, "user_id"
        ):
            logger.warning("Recreating index %s — missing user_id mapping", index_name)
            client.indices.delete(index=index_name)
            _wipe_local_document_storage()
            break

    results: dict[str, bool] = {}
    index_specs = [
        (settings.chunks_index, chunks_index_body()),
        (settings.documents_index, documents_index_body()),
    ]
    for index_name, body in index_specs:
        if client.indices.exists(index=index_name):
            results[index_name] = False
            continue
        client.indices.create(index=index_name, body=body)
        results[index_name] = True
    return results


def get_index_mapping(client: OpenSearch, index_name: str) -> dict[str, Any]:
    try:
        return client.indices.get_mapping(index=index_name)
    except NotFoundError as exc:
        raise ValueError(f"Index {index_name} does not exist") from exc
