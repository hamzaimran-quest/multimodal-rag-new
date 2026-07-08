"""OpenSearch bootstrap: indices, hybrid pipeline, and connectivity checks."""

import logging
import time
from typing import Any

from opensearchpy import OpenSearch
from opensearchpy.exceptions import ConnectionError as OSConnectionError

from app.config import settings
from app.opensearch.client import get_opensearch_client
from app.opensearch.indices import ensure_indices
from app.opensearch.pipelines import ensure_hybrid_search_pipeline

logger = logging.getLogger(__name__)

MAX_BOOTSTRAP_RETRIES = 30
RETRY_DELAY_SECONDS = 2


def wait_for_opensearch(client: OpenSearch | None = None) -> OpenSearch:
  """Block until OpenSearch responds to cluster health."""
  client = client or get_opensearch_client()
  last_error: Exception | None = None
  for attempt in range(1, MAX_BOOTSTRAP_RETRIES + 1):
    try:
      health = client.cluster.health()
      if health.get("status") in {"green", "yellow"}:
        logger.info("OpenSearch cluster healthy (status=%s)", health.get("status"))
        return client
    except OSConnectionError as exc:
      last_error = exc
      logger.debug("OpenSearch not ready (attempt %s/%s)", attempt, MAX_BOOTSTRAP_RETRIES)
    time.sleep(RETRY_DELAY_SECONDS)
  raise RuntimeError("OpenSearch did not become ready in time") from last_error


def bootstrap_opensearch(client: OpenSearch | None = None) -> dict[str, Any]:
  """Ensure indices and hybrid search pipeline exist."""
  client = client or wait_for_opensearch()
  indices_created = ensure_indices(client)
  pipeline_created = ensure_hybrid_search_pipeline(client)
  return {
    "indices": indices_created,
    "hybrid_pipeline_created": pipeline_created,
    "chunks_index": settings.chunks_index,
    "documents_index": settings.documents_index,
    "hybrid_search_pipeline": settings.hybrid_search_pipeline,
  }
