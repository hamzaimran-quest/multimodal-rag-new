"""Heuristics for when document search should reserve table chunk slots."""

from __future__ import annotations

import re

# Multi-word phrases — matched as substrings on normalized query text.
_COMPARISON_PHRASES: tuple[str, ...] = (
    "compare",
    "comparison",
    "compared",
    "comparing",
    "versus",
    " vs ",
    " vs.",
    " vs,",
    "difference between",
    "differences between",
    "year-over-year",
    "year on year",
    "year over year",
    "year-on-year",
    "quarter over quarter",
    "quarter-on-quarter",
    "side by side",
    "higher than",
    "lower than",
    "increased by",
    "decreased by",
    "growth rate",
    "rate of growth",
    "how much did",
    "how much has",
    "change from",
    "changed from",
    "change between",
    "changed between",
    "from last year",
    "from the prior year",
    "prior year",
    "previous year",
    "same period",
    "two-year",
    "three-year",
    "five-year",
    "multi-year",
)

# Single tokens — matched with word boundaries to reduce false positives.
_METRIC_TERMS: frozenset[str] = frozenset(
    {
        "revenue",
        "revenues",
        "sales",
        "turnover",
        "topline",
        "income",
        "profit",
        "profits",
        "profitable",
        "profitability",
        "earnings",
        "ebitda",
        "ebit",
        "ebt",
        "net",
        "gross",
        "margin",
        "margins",
        "operating",
        "cashflow",
        "liquidity",
        "dividend",
        "dividends",
        "eps",
        "roe",
        "roa",
        "roi",
        "roce",
        "assets",
        "liabilities",
        "equity",
        "debt",
        "capex",
        "opex",
        "expenditure",
        "expenditures",
        "expenses",
        "costs",
        "cost",
        "spend",
        "spending",
        "cogs",
        "arpu",
        "arr",
        "mrr",
        "gmv",
        "highlights",
        "highlight",
        "figures",
        "metrics",
        "kpi",
        "kpis",
        "breakdown",
        "segment",
        "segments",
        "regional",
        "region",
        "regions",
        "geography",
        "geographic",
        "fiscal",
        "fy",
        "yoy",
        "qoq",
        "mom",
        "billion",
        "millions",
        "million",
        "billions",
        "percent",
        "percentage",
        "margin",
    }
)

_REPORT_PHRASES: tuple[str, ...] = (
    "financial highlights",
    "key figures",
    "key metrics",
    "financial summary",
    "financial performance",
    "financial results",
    "annual results",
    "quarterly results",
    "segment revenue",
    "revenue by",
    "profit by",
    "sales by",
    "by region",
    "by segment",
    "by geography",
    "by business",
    "income statement",
    "balance sheet",
    "cash flow statement",
)

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_METRIC_TOKEN_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(term) for term in sorted(_METRIC_TERMS, key=len, reverse=True)) + r")\b"
)


def numeric_comparison_query_detected(query: str) -> bool:
    """True when the query likely needs tabular financial or comparison context."""
    normalized = f" {query.strip().lower()} "
    if not normalized.strip():
        return False

    for phrase in _COMPARISON_PHRASES:
        if phrase in normalized:
            return True

    for phrase in _REPORT_PHRASES:
        if phrase in normalized:
            return True

    if _METRIC_TOKEN_RE.search(normalized):
        return True

    years = _YEAR_RE.findall(normalized)
    if len(years) >= 2:
        return True

    # One explicit year plus any financial/metric language.
    if years and _METRIC_TOKEN_RE.search(normalized):
        return True

    return False


def should_reserve_table_slots(*, pdf_scope: bool, query: str) -> bool:
    """Gate for PDF_TABLE_SLOTS / merge_with_table_slot."""
    return pdf_scope or numeric_comparison_query_detected(query)
