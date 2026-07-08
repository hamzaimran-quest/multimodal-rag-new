"""Unit tests for text chunking."""

from app.ingestion.chunking import chunk_text, normalize_whitespace


def test_normalize_whitespace():
    assert normalize_whitespace("  hello   world\n\nfoo  ") == "hello world foo"


def test_chunk_text_short_input():
    assert chunk_text("one two three four five six seven eight") == [
        "one two three four five six seven eight"
    ]


def test_chunk_text_overlap():
    words = [f"w{i}" for i in range(20)]
    text = " ".join(words)
    chunks = chunk_text(text, max_words=10, overlap_words=2)
    assert len(chunks) >= 2
    assert chunks[0].startswith("w0")
    assert "w9" in chunks[0]
    assert chunks[1].startswith("w8")
