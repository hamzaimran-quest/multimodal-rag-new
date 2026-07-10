"""Render Chart.js configs via the QuickChart.io API."""

from __future__ import annotations

from typing import Any

from quickchart import QuickChart

from app.config import settings


def build_quickchart_url(config: dict[str, Any]) -> str:
    """Return a QuickChart image URL for a Chart.js config object."""
    qc = QuickChart()
    qc.width = settings.quickchart_width
    qc.height = settings.quickchart_height
    qc.version = "2"
    qc.config = config
    return qc.get_url()


def chartjs_config_from_data_spec(
    spec: dict[str, Any],
    *,
    chart_type_hint: str | None = None,
) -> dict[str, Any]:
    """Build a Chart.js config deterministically from {labels, series, title, chart_type}."""
    resolved_type = str(spec.get("chart_type") or chart_type_hint or "bar").strip().lower()
    if resolved_type not in {"bar", "line"}:
        resolved_type = "bar"

    labels = [str(label) for label in spec["labels"]]
    datasets = [
        {
            "label": str(entry["name"]).strip(),
            "data": [float(value) for value in entry["values"]],
        }
        for entry in spec["series"]
    ]
    title = str(spec.get("title") or "").strip()

    config: dict[str, Any] = {
        "type": resolved_type,
        "data": {"labels": labels, "datasets": datasets},
    }
    if title:
        config["options"] = {
            "plugins": {
                "title": {"display": True, "text": title},
            },
        }
    return config


def chart_title_from_config(config: dict[str, Any]) -> str | None:
    options = config.get("options")
    if not isinstance(options, dict):
        return None
    plugins = options.get("plugins")
    if not isinstance(plugins, dict):
        return None
    title = plugins.get("title")
    if not isinstance(title, dict):
        return None
    text = str(title.get("text") or "").strip()
    return text or None


def series_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive legacy series payload for UI fallbacks."""
    data = config.get("data") or {}
    labels = data.get("labels") or []
    datasets = data.get("datasets") or []
    series: list[dict[str, Any]] = []
    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        name = str(dataset.get("label") or "Series").strip() or "Series"
        values = dataset.get("data") or []
        series.append({"name": name, "values": [float(value) for value in values]})
    if not series:
        return []
    return series
