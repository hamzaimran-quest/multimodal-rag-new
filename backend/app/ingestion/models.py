"""Shared ingestion data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedChunk:
    content: str
    page_number: int
    chunk_type: str
    extraction_method: str
    bbox: list[float] | None = None
    image_path: str | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)
