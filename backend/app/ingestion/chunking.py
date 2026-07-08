"""Word-based text chunking with overlap."""

from app.config import settings


def chunk_text(
    text: str,
    max_words: int | None = None,
    overlap_words: int | None = None,
) -> list[str]:
    """Split text into overlapping word windows."""
    max_words = max_words or settings.chunk_max_words
    overlap_words = overlap_words or settings.chunk_overlap_words

    words = text.split()
    if not words:
        return []

    if len(words) <= max_words:
        return [" ".join(words)]

    chunks: list[str] = []
    step = max(max_words - overlap_words, 1)
    for start in range(0, len(words), step):
        window = words[start : start + max_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + max_words >= len(words):
            break

    return chunks


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())
