"""Standalone query rewriting for follow-ups — prior user queries only, aux LLM."""

from __future__ import annotations

import logging

import httpx

from app.config import settings
from app.llm.groq import GROQ_CHAT_COMPLETIONS_URL

logger = logging.getLogger(__name__)

QUERY_REWRITE_SYSTEM = """You rewrite the user's latest message into a standalone search query for document retrieval.

You receive prior user questions from this chat (oldest first), optionally the most recent assistant reply, and the latest user message. Combine them only when the latest message depends on earlier turns; otherwise return the latest message unchanged.

Rules:
- Output ONLY the standalone search query text.
- No quotes, labels, or explanation.
- Preserve specific names, numbers, and entities from prior questions or the latest assistant reply when resolving references (e.g. pronouns) in the latest message.
- If the latest message is already self-contained, output it as-is."""

_MAX_REWRITE_CHARS = 500


def _build_rewrite_user_prompt(
    prior_queries: list[str],
    user_query: str,
    last_assistant_reply: str | None = None,
) -> str:
    lines: list[str] = []
    if prior_queries:
        lines.append("Prior user questions:")
        for index, question in enumerate(prior_queries, start=1):
            lines.append(f"{index}. {question}")
        lines.append("")
    if last_assistant_reply:
        lines.append("Latest assistant reply:")
        lines.append(last_assistant_reply)
        lines.append("")
    lines.append(f"Latest message: {user_query}")
    return "\n".join(lines)


async def rewrite_query_for_retrieval(
    user_query: str,
    prior_queries: list[str] | None,
    last_assistant_reply: str | None = None,
) -> str:
    """
    Produce a standalone retrieval query from the latest message and prior chat context.

    Skips the LLM when rewrite is disabled or there is no prior user question or reply.
    """
    cleaned_query = user_query.strip()
    if not cleaned_query:
        return user_query

    prior = [q.strip() for q in (prior_queries or []) if q and q.strip()]
    last_reply = (last_assistant_reply or "").strip() or None
    if not settings.query_rewrite_enabled or (not prior and not last_reply):
        return user_query

    if not settings.groq_configured:
        return user_query

    messages = [
        {"role": "system", "content": QUERY_REWRITE_SYSTEM},
        {
            "role": "user",
            "content": _build_rewrite_user_prompt(prior, cleaned_query, last_reply),
        },
    ]
    payload = {
        "model": settings.query_rewrite_model,
        "messages": messages,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(GROQ_CHAT_COMPLETIONS_URL, headers=headers, json=payload)
            response.raise_for_status()
            rewritten = (response.json()["choices"][0]["message"].get("content") or "").strip()
            rewritten = rewritten.strip("\"'")
            if rewritten and len(rewritten) <= _MAX_REWRITE_CHARS:
                logger.info(
                    "QUERY_REWRITE model=%s prior_count=%s original=%r rewritten=%r",
                    settings.query_rewrite_model,
                    len(prior),
                    cleaned_query[:80],
                    rewritten[:80],
                )
                return rewritten
    except Exception:
        logger.warning(
            "QUERY_REWRITE failed model=%s query=%r",
            settings.query_rewrite_model,
            cleaned_query[:80],
            exc_info=True,
        )
    return user_query
