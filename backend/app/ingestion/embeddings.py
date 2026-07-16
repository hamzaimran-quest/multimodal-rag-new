"""FastEmbed (ONNX) embedding helpers with explicit L2 normalization."""

from __future__ import annotations

import logging
import math
import threading
from functools import lru_cache

from fastembed import TextEmbedding

from app.config import settings

logger = logging.getLogger(__name__)

# Guards first-ever model construction: fastembed downloads model files via
# huggingface_hub's threaded snapshot_download, and racing two of those
# downloads from concurrent request/ingestion threads corrupts tqdm's shared
# lock (AttributeError: 'tqdm' object has no attribute '_lock'). lru_cache
# alone doesn't prevent this — it only serializes cache reads/writes, not the
# wrapped call itself, so concurrent cache misses can both run through.
# The model loads lazily on first use (first query or first ingestion) and
# is cached in-process thereafter via lru_cache, so this cost is paid once.
_model_lock = threading.Lock()

# FastEmbed registry name (maps from short env value all-MiniLM-L6-v2)
FASTEMBED_MODEL_NAMES: dict[str, str] = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}

UNIT_NORM_TOLERANCE = 1e-4


def resolve_fastembed_model_name(model: str) -> str:
    return FASTEMBED_MODEL_NAMES.get(model, model)


@lru_cache(maxsize=1)
def _load_embedding_model() -> TextEmbedding:
    model_name = resolve_fastembed_model_name(settings.embedding_model)
    logger.info("Loading FastEmbed model: %s", model_name)
    return TextEmbedding(model_name=model_name)


def get_embedding_model() -> TextEmbedding:
    with _model_lock:
        return _load_embedding_model()


def l2_normalize(vector: list[float]) -> tuple[list[float], float]:
    """L2-normalize to unit length; return (vector, norm_before_normalize)."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector, 0.0
    return [value / norm for value in vector], norm


def vector_l2_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def assert_unit_norm(norm: float, *, index: int, tolerance: float = UNIT_NORM_TOLERANCE) -> None:
    if abs(norm - 1.0) > tolerance:
        raise ValueError(
            f"Vector at index {index} is not unit-normalized: L2 norm={norm:.6f} "
            f"(expected 1.0 ± {tolerance})"
        )


def assert_unit_vectors(vectors: list[list[float]], *, tolerance: float = UNIT_NORM_TOLERANCE) -> None:
    """Raise ValueError if any vector is not unit-normalized within tolerance."""
    for index, vector in enumerate(vectors):
        assert_unit_norm(vector_l2_norm(vector), index=index, tolerance=tolerance)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    model = get_embedding_model()
    raw_vectors = [list(vector) for vector in model.embed(texts)]

    # FastEmbed normalizes many models internally; re-normalize explicitly so ingest
    # and query paths always match prior sentence-transformers behavior.
    vectors: list[list[float]] = []
    for index, raw_vector in enumerate(raw_vectors):
        normalized, source_norm = l2_normalize(list(raw_vector))
        if source_norm == 0.0:
            raise ValueError(f"Vector at index {index} has zero norm")
        vectors.append(normalized)
    return vectors
