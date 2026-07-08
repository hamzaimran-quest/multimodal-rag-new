"""Hybrid search pipeline setup for BM25 + k-NN score fusion."""

from typing import Any

from opensearchpy import OpenSearch

from app.config import settings


def hybrid_pipeline_body() -> dict[str, Any]:
    return {
        "description": "Normalize and combine BM25 and k-NN scores for hybrid search",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": [0.5, 0.5]},
                    },
                }
            }
        ],
    }


def ensure_hybrid_search_pipeline(client: OpenSearch) -> bool:
    """Create the hybrid search pipeline if missing. Returns True if created."""
    pipeline_id = settings.hybrid_search_pipeline
    try:
        client.transport.perform_request(
            "GET",
            f"/_search/pipeline/{pipeline_id}",
        )
        return False
    except Exception:
        pass

    client.transport.perform_request(
        "PUT",
        f"/_search/pipeline/{pipeline_id}",
        body=hybrid_pipeline_body(),
    )
    return True
