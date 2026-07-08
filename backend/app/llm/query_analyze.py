"""History-aware query analysis: reformulation + visual intent in one call.

Used for follow-up turns where the current message references earlier context
(e.g. "show an image of him"). A single fast Groq call both rewrites the message
into a standalone search query and classifies visual intent, so retrieval and the
image gate operate on the resolved query rather than an unresolved pronoun.
Fail-open: on any error the original query is kept and intent defaults to none.
"""

from __future__ import annotations

import json
import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"

_VALID_INTENTS = {"required", "optional", "none"}

_SYSTEM_PROMPT = """You rewrite a user's latest message into a standalone search query and classify visual intent.

You are given the recent conversation and the latest user message. The latest message may rely on earlier context (pronouns like "he/she/it/his", or omitted subjects).

Return STRICT JSON: {"standalone_query": "...", "visual_intent": "required"|"optional"|"none"}

Rules:
- "standalone_query": rewrite the latest message into a fully self-contained query by resolving references using the conversation. When a pronoun refers to an entity, KEEP the descriptive role/title words used earlier (e.g. "chairman", "CEO", "the report") and ADD the specific name as extra context — do NOT replace the role with only the name. Example: if earlier the topic was "chairman of Huawei" and the answer named "Liang Hua", rewrite "show an image of him" as "image of the chairman of Huawei Liang Hua" (keep both role and name). If the latest message is already self-contained, return it essentially unchanged. Keep it concise; do not add facts not present in the conversation.
- "visual_intent": whether the latest message explicitly asks to SEE a visual asset (photo, portrait, figure, diagram, chart, screenshot, logo).
  - "required": explicit request to view an image/figure/chart/photo, including follow-ups like "show an image of him".
  - "optional": ambiguous.
  - "none": a normal factual/textual question.

Output JSON only."""


def _format_history(history: list[dict]) -> str:
    lines: list[str] = []
    for turn in history:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def analyze_query(history: list[dict], query: str) -> dict:
    """Return {"standalone_query", "visual_intent", "confidence"}.

    Falls back to the original query + ``none`` intent on any failure.
    """
    fallback = {"standalone_query": query, "visual_intent": "none", "confidence": 0.0}
    if not settings.image_intent_enabled:
        return fallback
    if not settings.groq_api_key or settings.groq_api_key == "your_groq_api_key_here":
        return fallback

    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }
    user_content = (
        f"Conversation:\n{_format_history(history)}\n\n"
        f"Latest user message:\n{query}"
    )
    payload = {
        "model": settings.image_intent_model,
        "temperature": 0.0,
        "max_tokens": 200,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.post(
                GROQ_CHAT_COMPLETIONS_URL, headers=headers, json=payload
            )
            response.raise_for_status()
            body = response.json()
            parsed = json.loads(body["choices"][0]["message"]["content"])
    except Exception:
        logger.warning("Query analysis failed; using original query", exc_info=True)
        return fallback

    standalone = str(parsed.get("standalone_query") or "").strip() or query
    intent = str(parsed.get("visual_intent", "none")).strip().lower()
    if intent not in _VALID_INTENTS:
        intent = "none"
    return {"standalone_query": standalone, "visual_intent": intent, "confidence": 0.0}
