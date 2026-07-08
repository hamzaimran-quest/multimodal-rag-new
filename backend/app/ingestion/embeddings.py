"""FastEmbed (ONNX) embedding helpers with explicit L2 normalization."""

from __future__ import annotations

import logging
import math
from functools import lru_cache

from fastembed import TextEmbedding

from app.config import settings

logger = logging.getLogger(__name__)

# FastEmbed registry name (maps from short env value all-MiniLM-L6-v2)
FASTEMBED_MODEL_NAMES: dict[str, str] = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}

UNIT_NORM_TOLERANCE = 1e-4


def resolve_fastembed_model_name(model: str) -> str:
    return FASTEMBED_MODEL_NAMES.get(model, model)


@lru_cache(maxsize=1)
def get_embedding_model() -> TextEmbedding:
    model_name = resolve_fastembed_model_name(settings.embedding_model)
    logger.info("Loading FastEmbed model: %s", model_name)
    return TextEmbedding(model_name=model_name)


def l2_normalize(vector: list[float]) -> list[float]:
    """L2-normalize to unit length (matches sentence-transformers normalize_embeddings=True)."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def vector_l2_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def assert_unit_vectors(vectors: list[list[float]], *, tolerance: float = UNIT_NORM_TOLERANCE) -> None:
    """Raise ValueError if any vector is not unit-normalized within tolerance."""
    for index, vector in enumerate(vectors):
        norm = vector_l2_norm(vector)
        if abs(norm - 1.0) > tolerance:
            raise ValueError(
                f"Vector at index {index} is not unit-normalized: L2 norm={norm:.6f} "
                f"(expected 1.0 ± {tolerance})"
            )


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    model = get_embedding_model()
    raw_vectors = [list(vector) for vector in model.embed(texts)]

    # FastEmbed normalizes many models internally; re-normalize explicitly so ingest
    # and query paths always match prior sentence-transformers behavior.
    vectors = [l2_normalize(vector) for vector in raw_vectors]
    assert_unit_vectors(vectors)
    return vectors
