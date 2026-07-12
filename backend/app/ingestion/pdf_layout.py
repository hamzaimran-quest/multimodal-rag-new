"""Generic per-page reading-order detection for pdfplumber word lists."""

from __future__ import annotations

MIN_WORDS_FOR_LAYOUT = 15
HISTOGRAM_BINS = 30
HISTOGRAM_MARGIN_RATIO = 0.2
VALLEY_TO_PEAK_RATIO = 0.35
MIN_SIDE_WORD_RATIO = 0.12


def _word_x_center(word: dict) -> float:
    return (float(word["x0"]) + float(word["x1"])) / 2.0


def _sort_words_top_down(words: list[dict]) -> list[dict]:
    """Sort words top-to-bottom, then left-to-right within a column band."""
    return sorted(words, key=lambda w: (round(float(w["top"]), 1), float(w["x0"])))


def detect_column_gutter(words: list[dict], page_width: float) -> float | None:
    """Return the x-coordinate of a column gutter, or None for single-column pages."""
    if len(words) < MIN_WORDS_FOR_LAYOUT or page_width <= 0:
        return None

    counts = [0] * HISTOGRAM_BINS
    for word in words:
        center = _word_x_center(word)
        bucket = min(HISTOGRAM_BINS - 1, max(0, int(center / page_width * HISTOGRAM_BINS)))
        counts[bucket] += 1

    smoothed = [counts[0]]
    for index in range(1, HISTOGRAM_BINS - 1):
        smoothed.append((counts[index - 1] + counts[index] + counts[index + 1]) / 3.0)
    smoothed.append(float(counts[-1]))

    search_lo = int(HISTOGRAM_BINS * HISTOGRAM_MARGIN_RATIO)
    search_hi = int(HISTOGRAM_BINS * (1.0 - HISTOGRAM_MARGIN_RATIO))
    if search_hi <= search_lo:
        return None

    valley_index = min(range(search_lo, search_hi), key=lambda i: smoothed[i])
    valley = smoothed[valley_index]
    peak = max(smoothed[search_lo:search_hi])
    if peak <= 0 or valley / peak > VALLEY_TO_PEAK_RATIO:
        return None

    gutter_x = (valley_index + 0.5) * page_width / HISTOGRAM_BINS
    left_count = sum(1 for word in words if _word_x_center(word) < gutter_x)
    right_count = len(words) - left_count
    if left_count < len(words) * MIN_SIDE_WORD_RATIO:
        return None
    if right_count < len(words) * MIN_SIDE_WORD_RATIO:
        return None
    return gutter_x


def column_bands_for_reading(words: list[dict], page_width: float) -> list[list[dict]]:
    """Split words into ordered column bands (one band for single-column pages)."""
    if not words:
        return []

    gutter_x = detect_column_gutter(words, page_width)
    if gutter_x is None:
        return [_sort_words_top_down(words)]

    left_words = [word for word in words if _word_x_center(word) < gutter_x]
    right_words = [word for word in words if _word_x_center(word) >= gutter_x]
    bands: list[list[dict]] = []
    if left_words:
        bands.append(_sort_words_top_down(left_words))
    if right_words:
        bands.append(_sort_words_top_down(right_words))
    return bands


def reorder_words_for_reading(words: list[dict], page_width: float) -> list[dict]:
    """Reorder words for human reading order on single- or two-column pages."""
    ordered: list[dict] = []
    for band in column_bands_for_reading(words, page_width):
        ordered.extend(band)
    return ordered
