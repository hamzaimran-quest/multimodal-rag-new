"""History-aware query writer for follow-up retrieval.

Rewrites ambiguous follow-up messages into standalone search queries using recent
conversation context (pronouns, demonstratives, omitted subjects, continuations).
Also classifies visual intent in the same Groq call on follow-up turns.

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

_SYSTEM_PROMPT = """You are a query writer for a document search system.

You receive the recent conversation (user and assistant turns) and the user's latest message. The latest message is often ambiguous on its own — it may depend on earlier turns. Your job is to rewrite it into a clear, detailed, self-contained search query that retrieval can use without seeing the conversation.

Return STRICT JSON: {"standalone_query": "...", "visual_intent": "required"|"optional"|"none"}

## standalone_query

Rewrite the latest user message into one concise but specific search query. Use the full conversation to resolve every kind of reference, not only pronouns:

- **Pronouns** (he/she/it/him/her/them/they): resolve to the entity discussed earlier; keep role/title words AND names when known (e.g. chairman + Liang Hua).
- **Demonstratives** (this/that/these/those/the above/the same): resolve to the topic, metric, table, region, product, or comparison from prior turns (e.g. "compare these with 2024" after revenue discussion → "compare Huawei annual revenue 2024 vs 2025").
- **Omitted subjects** ("compare with 2024", "what about Europe?", "and operating profit?"): supply the entity and metric implied by context.
- **Continuations** ("same for cloud", "what about 2023?", "break that down by region"): carry forward the subject from earlier Q&A.
- **Implicit comparisons** ("vs last year", "year over year"): include the concrete years and metrics when stated in the conversation.

Rules:
- Include specific names, years, metrics, regions, and document topics from the conversation when they clarify the request.
- Do NOT invent facts that are not supported by the conversation.
- If the latest message is already fully self-contained, return it essentially unchanged.
- Write for search retrieval: prefer concrete nouns (revenue, chairman, region, financial highlights) over vague words (these, it, that).
- One line; no preamble.

Examples:
- Prior: user asked chairman of Huawei, assistant named Liang Hua. Latest: "show an image of him" → "image photo of Huawei chairman Liang Hua"
- Prior: user asked Huawei finances, assistant gave 2025 revenue. Latest: "compare these with 2024" → "compare Huawei annual revenue 2024 vs 2025 financial highlights"
- Prior: discussed Asia-Pacific revenue. Latest: "what about EMEA?" → "Huawei revenue Europe Middle East Africa EMEA 2025"

## visual_intent

Whether the latest message explicitly asks to SEE a visual asset (photo, portrait, figure, diagram, chart, screenshot, logo):
- "required": explicit request to view an image/figure/chart/photo, including visual follow-ups.
- "optional": ambiguous.
- "none": normal factual/textual question.

Output JSON only."""


def _format_history(history: list[dict]) -> str:
    lines: list[str] = []
    for turn in history:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _parse_analysis_response(parsed: dict, *, original_query: str) -> dict:
    standalone = str(parsed.get("standalone_query") or "").strip() or original_query
    intent = str(parsed.get("visual_intent", "none")).strip().lower()
    if intent not in _VALID_INTENTS:
        intent = "none"
    return {"standalone_query": standalone, "visual_intent": intent, "confidence": 0.0}


async def analyze_query(history: list[dict], query: str) -> dict:
    """Return {"standalone_query", "visual_intent", "confidence"}.

    Falls back to the original query + ``none`` intent on any failure.
    """
    fallback = {"standalone_query": query, "visual_intent": "none", "confidence": 0.0}
    if not settings.query_rewrite_enabled:
        return fallback
    if not settings.groq_api_key or settings.groq_api_key == "your_groq_api_key_here":
        return fallback

    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }
    user_content = (
        f"Conversation (oldest first, up to {settings.chat_history_turns} turns):\n"
        f"{_format_history(history)}\n\n"
        f"Latest user message:\n{query}"
    )
    payload = {
        "model": settings.query_rewrite_model,
        "temperature": 0.0,
        "max_tokens": 300,
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
        logger.warning("Query rewrite failed; using original query", exc_info=True)
        return fallback

    result = _parse_analysis_response(parsed, original_query=query)
    if result["standalone_query"] != query:
        logger.info(
            "QUERY_REWRITE original=%r standalone=%r",
            query[:80],
            result["standalone_query"][:120],
        )
    return result
