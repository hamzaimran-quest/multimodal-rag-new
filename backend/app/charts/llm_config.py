"""Use an aux LLM to extract chart data from tables; Chart.js is built in code."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

import httpx

from app.config import settings
from app.llm.groq import GROQ_CHAT_COMPLETIONS_URL

logger = logging.getLogger(__name__)

ChartTypeHint = Literal["bar", "line"] | None

_SYSTEM_PROMPT = """You extract chart-ready data from document table excerpts.

Output ONLY a single JSON object (no markdown fences, no commentary).

Schema:
{
  "chart_type": "bar" or "line",
  "title": "short descriptive title",
  "labels": ["category1", "category2", ...],
  "series": [{"name": "Series name", "values": [number, number, ...]}]
}

Rules:
- Use only numbers explicitly present in the table. Do not invent values.
- labels: x-axis categories (years, regions, segments, etc.) — max 12.
- series: max 8 entries; each values array MUST have exactly len(labels) numbers.
- Strip currency symbols and commas; use plain floats. Skip % / YoY columns unless the user asks for them.
- Exclude "Total" rows unless the user asks for totals.
- Pick labels + series orientation that best answers the user's chart request.

Example (regional revenue by year — one series per region):
{
  "chart_type": "bar",
  "title": "Regional revenue",
  "labels": ["2024", "2025"],
  "series": [
    {"name": "China", "values": [615264, 616249]},
    {"name": "EMEA", "values": [148355, 161356]}
  ]
}"""

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _strip_json_fences(text: str) -> str:
    match = _JSON_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _validate_chart_data_spec(spec: dict[str, Any]) -> str | None:
    chart_type = str(spec.get("chart_type", "bar")).strip().lower()
    if chart_type not in {"bar", "line"}:
        return "Chart type must be bar or line."

    labels = spec.get("labels")
    series = spec.get("series")
    if not isinstance(labels, list) or not labels:
        return "Missing labels."
    if not isinstance(series, list) or not series:
        return "Missing series."
    if len(labels) > 12:
        return "Too many labels."
    if len(series) > 8:
        return "Too many series."

    width = len(labels)
    for index, entry in enumerate(series):
        if not isinstance(entry, dict):
            return f"Series {index} is invalid."
        name = str(entry.get("name") or "").strip()
        if not name:
            return f"Series {index} is missing a name."
        values = entry.get("values")
        if not isinstance(values, list) or len(values) != width:
            return f"Series {index} length does not match labels."
        for value in values:
            if not isinstance(value, (int, float)):
                return f"Series {index} contains non-numeric values."

    return None


def extract_chart_data_spec(
    user_query: str,
    table_markdown: str,
    *,
    chart_type: ChartTypeHint = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Call the aux LLM to produce {labels, series} from a table excerpt."""
    if not settings.groq_configured:
        return None, "Chart LLM is not configured."

    table = table_markdown.strip()
    if not table:
        return None, "Table content is empty."

    type_hint = chart_type or "auto"
    user_content = (
        f"User chart request: {user_query.strip()}\n"
        f"Requested chart type: {type_hint}\n\n"
        f"Table excerpt:\n{table[: settings.chart_table_max_chars]}"
    )

    payload = {
        "model": settings.chart_llm_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=45.0) as client:
            response = client.post(GROQ_CHAT_COMPLETIONS_URL, headers=headers, json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"].get("content") or ""
    except Exception as exc:
        logger.warning("CHART_LLM request failed query=%r", user_query[:80], exc_info=True)
        return None, f"Chart configuration LLM failed: {exc}"

    try:
        parsed = json.loads(_strip_json_fences(content))
    except json.JSONDecodeError:
        logger.warning("CHART_LLM invalid_json preview=%r", content[:200])
        return None, "Chart LLM returned invalid JSON."

    if not isinstance(parsed, dict):
        return None, "Chart LLM returned a non-object JSON value."

    spec_json = json.dumps(parsed, ensure_ascii=False)
    logger.info(
        "CHART_LLM raw_spec query=%r model=%s spec=%s",
        user_query[:80],
        settings.chart_llm_model,
        spec_json[:2000],
    )

    error = _validate_chart_data_spec(parsed)
    if error:
        labels = parsed.get("labels") if isinstance(parsed.get("labels"), list) else []
        series_lens = [
            len(entry.get("values") or [])
            for entry in (parsed.get("series") or [])
            if isinstance(entry, dict)
        ]
        logger.info(
            "CHART_LLM spec_rejected reason=%s label_count=%s series_value_lens=%s spec=%s",
            error,
            len(labels),
            series_lens,
            spec_json[:2000],
        )
        return None, error

    logger.info(
        "CHART_LLM ok model=%s type=%s labels=%s series=%s",
        settings.chart_llm_model,
        parsed.get("chart_type"),
        len(parsed.get("labels", [])),
        len(parsed.get("series", [])),
    )
    return parsed, None
