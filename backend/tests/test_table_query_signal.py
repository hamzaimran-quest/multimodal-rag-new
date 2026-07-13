"""Tests for table-slot query signal detection."""

from __future__ import annotations

import pytest

from app.retrieval.table_query_signal import (
    numeric_comparison_query_detected,
    should_reserve_table_slots,
)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("compare revenue 2025 and 2024", True),
        ("what are the financial highlights", True),
        ("Huawei profit by segment", True),
        ("year-over-year sales growth", True),
        ("operating margin in FY2024", True),
        ("how much did net income change", True),
        ("segment revenue by region", True),
        ("cash flow statement figures", True),
        ("ebitda versus last year", True),
        ("2023 and 2024 results", True),
        ("dividend per share", True),
        ("balance sheet assets and liabilities", True),
        ("who is the chairwoman", False),
        ("summarize the introduction", False),
        ("what does the company do", False),
        ("show me the logo", False),
    ],
)
def test_numeric_comparison_query_detected(query: str, expected: bool) -> None:
    assert numeric_comparison_query_detected(query) is expected


def test_should_reserve_table_slots_pdf_scope_without_signal() -> None:
    assert should_reserve_table_slots(pdf_scope=True, query="who is the chairwoman") is True


def test_should_reserve_table_slots_query_signal_without_pdf_scope() -> None:
    assert should_reserve_table_slots(pdf_scope=False, query="compare revenue 2024 and 2025") is True


def test_should_reserve_table_slots_neither() -> None:
    assert should_reserve_table_slots(pdf_scope=False, query="who is the chairwoman") is False
