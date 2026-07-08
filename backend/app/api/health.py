"""Health and readiness endpoints."""

from typing import Any

from fastapi import APIRouter, Request
from opensearchpy.exceptions import ConnectionError as OSConnectionError

from app.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/config")
async def client_config() -> dict[str, Any]:
    """Public, non-sensitive client settings (e.g. viewer tunables)."""
    return {"pdf_viewer_page_window": settings.pdf_viewer_page_window}


@router.get("/ready")
async def ready(request: Request) -> dict[str, Any]:
    """Readiness: OpenSearch reachable and indices exist."""
    client = request.app.state.opensearch
    checks: dict[str, Any] = {
        "opensearch": "unknown",
        "chunks_index": False,
        "documents_index": False,
        "hybrid_pipeline": False,
        "groq_configured": settings.groq_configured,
    }
    try:
        health = client.cluster.health()
        checks["opensearch"] = health.get("status", "unknown")
        checks["chunks_index"] = client.indices.exists(index=settings.chunks_index)
        checks["documents_index"] = client.indices.exists(index=settings.documents_index)
        try:
            client.transport.perform_request(
                "GET",
                f"/_search/pipeline/{settings.hybrid_search_pipeline}",
            )
            checks["hybrid_pipeline"] = True
        except Exception:
            checks["hybrid_pipeline"] = False
    except OSConnectionError:
        checks["opensearch"] = "unreachable"

    ready = (
        checks["opensearch"] in {"green", "yellow"}
        and checks["chunks_index"]
        and checks["documents_index"]
        and checks["hybrid_pipeline"]
    )
    return {"ready": ready, "checks": checks}
