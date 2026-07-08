"""Tiny, fast Groq classifier for explicit visual intent.

Runs in parallel with embedding + hybrid retrieval (never blocks it) and only
decides whether the user is explicitly asking to *see* something visual — a
photo, figure, chart, diagram, or screenshot. It does NOT judge whether an image
would merely be "helpful"; implicit relevance is handled by proximity attachment.
Fail-closed: any error or misconfiguration yields ``none``.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"

VisualIntent = Literal["required", "optional", "none"]
_VALID_INTENTS = {"required", "optional", "none"}

_SYSTEM_PROMPT = """You classify whether a user's question explicitly asks to SEE a visual asset.

Visual assets are photos, portraits, figures, diagrams, charts, screenshots, logos, or illustrations.

Return STRICT JSON: {"visual_intent": "required"|"optional"|"none", "confidence": 0.0-1.0}

- "required": the user explicitly wants to view an image/figure/chart/photo. Examples: "show me the chart", "what does the chairman look like", "display the diagram", "show the org chart", "picture of the CEO".
- "optional": borderline phrasing where an image might be requested but it is ambiguous.
- "none": a normal factual/textual question, even if an image happens to exist. Examples: "who is the chairman", "what was revenue in 2024", "summarize the report".

Only classify intent to SEE something. Do not judge whether an image would be helpful. Output JSON only."""

_DEFAULT: dict = {"visual_intent": "none", "confidence": 0.0}


async def classify_visual_intent(query: str) -> dict:
    """Return {"visual_intent", "confidence"}; ``none`` on any failure."""
    if not settings.image_intent_enabled:
        return dict(_DEFAULT)
    if not settings.groq_api_key or settings.groq_api_key == "your_groq_api_key_here":
        return dict(_DEFAULT)

    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.image_intent_model,
        "temperature": 0.0,
        "max_tokens": 60,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.post(
                GROQ_CHAT_COMPLETIONS_URL, headers=headers, json=payload
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
    except Exception:
        logger.warning("Visual intent classification failed; defaulting to none", exc_info=True)
        return dict(_DEFAULT)

    intent = str(parsed.get("visual_intent", "none")).strip().lower()
    if intent not in _VALID_INTENTS:
        intent = "none"
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {"visual_intent": intent, "confidence": confidence}
