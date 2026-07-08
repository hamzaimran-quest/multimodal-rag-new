"""FastEmbed parity and normalization tests."""

from __future__ import annotations

import pytest

from app.config import settings
from app.ingestion.embeddings import (
    UNIT_NORM_TOLERANCE,
    assert_unit_vectors,
    embed_texts,
    l2_normalize,
    resolve_fastembed_model_name,
    vector_l2_norm,
)


def test_resolve_fastembed_model_name():
    assert (
        resolve_fastembed_model_name("all-MiniLM-L6-v2")
        == "sentence-transformers/all-MiniLM-L6-v2"
    )


def test_l2_normalize_unit_length():
    vector = l2_normalize([3.0, 4.0])
    assert vector == [0.6, 0.8]
    assert abs(vector_l2_norm(vector) - 1.0) < 1e-9


@pytest.mark.slow
def test_embed_texts_dimension_and_unit_norm():
    texts = ["revenue growth", "operating profit", "Five-Year Financial Highlights"]
    vectors = embed_texts(texts)

    assert len(vectors) == 3
    for vector in vectors:
        assert len(vector) == settings.embedding_dimension
        assert abs(vector_l2_norm(vector) - 1.0) <= UNIT_NORM_TOLERANCE


@pytest.mark.slow
def test_embed_texts_deterministic_for_same_input():
    text = "Huawei annual report revenue CNY Million"
    first = embed_texts([text])[0]
    second = embed_texts([text])[0]
    assert first == second


@pytest.mark.slow
def test_embed_texts_semantic_parity():
    """Related financial phrases should be closer than unrelated text (cosine ~ dot on unit vectors)."""
    anchor = embed_texts(["revenue growth financial highlights"])[0]
    related = embed_texts(["annual revenue increased year over year"])[0]
    unrelated = embed_texts(["the cat sat on the mat in the garden"])[0]

    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert dot(anchor, related) > dot(anchor, unrelated)


@pytest.mark.slow
def test_assert_unit_vectors_rejects_non_unit():
    bad = [1.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="not unit-normalized"):
        assert_unit_vectors([bad])
