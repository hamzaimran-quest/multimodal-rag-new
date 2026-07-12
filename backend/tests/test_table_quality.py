"""Unit tests for generic table quality checks."""

from app.ingestion.pdf_tables import (
    _infer_label_column,
    _move_label_column_to_front,
    _semantic_label_loss_detected,
    _should_short_circuit_extract,
)
from app.ingestion.tables import column_misalignment_ratio


def test_column_misalignment_ratio_detects_inconsistent_rows():
    rows = [
        ["Metric", "2025", "2024", "2023"],
        ["Revenue", "100", "90", "80"],
        ["Operating margin", "11.0%", "9.2%"],  # missing one cell
        ["Net profit", "70", "60", "50"],
    ]
    ratio = column_misalignment_ratio(rows)
    assert ratio > 0.15


def test_infer_label_column_by_content_type():
    rows = [
        ["2025", "Metric", "2024"],
        ["100", "Revenue", "90"],
        ["11.0%", "Operating margin", "9.2%"],
        ["70", "Net profit", "60"],
    ]
    label_col = _infer_label_column(rows)
    assert label_col == 1


def test_move_label_column_to_front():
    rows = [
        ["2025", "Metric", "2024"],
        ["100", "Revenue", "90"],
    ]
    moved = _move_label_column_to_front(rows, 1)
    assert moved[0][0] == "Metric"
    assert moved[1][0] == "Revenue"


def test_semantic_label_loss_detected_when_label_cells_are_empty():
    rows = [
        ["", "2025", "2024", "2023"],
        ["", "(USD Million)", "(CNY Million)", "(CNY Million)"],
        [None, "126,018", "880,941", "862,072"],
        [None, "13,867", "96,937", "79,361"],
        [None, "11.0%", "11.0%", "9.2%"],
        [None, "9,732", "68,036", "62,574"],
    ]
    semantic_loss, label_empty_ratio, numeric_ratio = _semantic_label_loss_detected(rows, label_col=0)
    assert semantic_loss is True
    assert label_empty_ratio >= 0.6
    assert numeric_ratio >= 0.6


def test_semantic_label_loss_not_detected_for_well_labeled_table():
    rows = [
        ["Metric", "2025", "2024", "2023"],
        ["Revenue", "126,018", "880,941", "862,072"],
        ["Operating margin", "11.0%", "11.0%", "9.2%"],
        ["Net profit", "9,732", "68,036", "62,574"],
        ["Liability ratio", "55.0%", "55.0%", "57.8%"],
    ]
    semantic_loss, label_empty_ratio, _ = _semantic_label_loss_detected(rows, label_col=0)
    assert semantic_loss is False
    assert label_empty_ratio < 0.6


def test_should_short_circuit_only_for_empty_labels_with_numeric_grid():
    assert _should_short_circuit_extract(
        {"label_empty_ratio": 1.0, "numeric_data_ratio": 1.0, "semantic_label_loss": True}
    )
    assert not _should_short_circuit_extract(
        {"label_empty_ratio": 1.0, "numeric_data_ratio": 0.0, "semantic_label_loss": False}
    )

