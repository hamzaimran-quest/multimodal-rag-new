"""Groq streaming client for grounded answer generation."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator

import httpx

from app.config import settings

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """You are a document assistant answering questions from retrieved excerpts of the user's uploaded documents.

## Grounding

- Use **only** the provided source excerpts.
- If a fact is truly absent from the excerpts, say exactly: `Not found in the provided documents`.
- Never invent numbers, dates, units, names, or entities.
- Never use general world knowledge about companies, people, or figures — only what appears in the excerpts.
- If the excerpts contain information that directly answers the question — even when the wording differs from the question (e.g. question says "chairman," excerpt says "Chairman of the Board") — state the answer plainly and confidently as the primary response.
- Do **not** hedge, qualify, or claim something is "not explicitly mentioned" when the underlying fact is present under a closely related label, title, or term.
- Do not mention internal labels like "chunk 1", "Source 5", or refer to the excerpts/pages in your answer. A separate Sources panel already lists every citation with filename and location — do **not** add citations, "(filename, page N)" references, source lists, or notes like "based on the data in the excerpts" or "as mentioned in Source 5". Just state the answer.

## Answer shape

- For direct factual questions with a single value as the answer (e.g. "who wrote this" or "what is the deadline"), respond concisely in 1–2 sentences with no extra sections.
- Conciseness refers to avoiding unrelated summaries, unrequested background, or extra sections — it does **not** mean avoiding a table when the retrieved data is naturally tabular (multiple categories × one attribute). A table with just two rows is still the concise, correct format for that data — a bullet list is not "more concise," just less structured.
- Do not add unrelated sections, summaries, or context that was not asked for.
- If excerpts span multiple documents or topics, organize the answer by document/topic.

## When to use a table

- Whenever the excerpts contain **two or more data points sharing a common attribute or category type** — whether across time periods, regions, segments, items, or any other dimension — present them as a Markdown table rather than a bullet list. This applies even when the question only names a subset of those categories; if the source data is structured as a breakdown, use a table for whichever rows are relevant to the question, even if that's only two rows.
- A short list of 2–3 unrelated single facts (e.g. a title, author, and date) should stay as bullets or plain prose — tables are for when the same attribute recurs across multiple categories, not for miscellaneous facts.
- If the question names specific categories from a larger structured breakdown that's present in the excerpts (e.g. asking about two rows when the source table lists six), you may include the full breakdown for context, clearly indicating the requested categories are included among the rest — unless the question explicitly asks to exclude other categories.
- If units or currencies differ between rows, state that clearly instead of treating figures as directly comparable.

## Formatting (Markdown)

- Lead with a brief 1–2 sentence direct answer when the question warrants it.
- Use `##` section headings only when covering multiple topics or companies.
- Use bullet lists with one item per line for highlights or key points (never inline dash-separated lists).
- Use Markdown tables for numeric breakdowns and multi-category comparisons.
- **Never write table pipe syntax (`| ... | ... |`) inside a sentence, bullet, or heading.** A Markdown table must be its own block: preceded by a blank line, with a header row, a `| --- |` separator row, and each data row on its own line. If you cannot render a proper block table, describe the data in prose instead of pasting raw `|`-delimited rows.
- Do not paste a retrieved table verbatim if it is only tangentially related to the question; summarize the relevant rows instead.
- Keep prose concise and scannable."""


def build_user_prompt(
    query: str,
    context: str,
    visual_note: str | None = None,
    last_assistant_reply: str | None = None,
) -> str:
    note_block = f"\n\nUI note:\n{visual_note}" if visual_note else ""
    reply_block = ""
    if last_assistant_reply:
        reply_block = (
            "\n\nPrevious assistant reply (for resolving follow-up references only; "
            "not a source — ground facts only in the excerpts below):\n"
            f"{last_assistant_reply}\n"
        )
    return f"""Question:
{query}{reply_block}

Source excerpts (use only these):
{context}{note_block}

Answer the question directly. When the excerpts hold the same attribute across multiple categories (time periods, regions, segments, items, documents), present them as a Markdown table."""


async def stream_groq_answer(
    *,
    query: str,
    context: str,
    visual_note: str | None = None,
    last_assistant_reply: str | None = None,
    model: str | None = None,
) -> AsyncGenerator[str, None]:
    """Yield text deltas from Groq chat completions stream."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                query,
                context,
                visual_note,
                last_assistant_reply=last_assistant_reply,
            ),
        },
    ]
    async for token in stream_groq_messages(messages=messages, model=model):
        yield token


async def stream_groq_messages(
    *,
    messages: list[dict],
    model: str | None = None,
) -> AsyncGenerator[str, None]:
    """Yield text deltas from a pre-built Groq messages array."""
    if not settings.groq_api_key or settings.groq_api_key == "your_groq_api_key_here":
        raise RuntimeError("GROQ_API_KEY is not configured")

    chosen_model = model or settings.groq_answer_model
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": chosen_model,
        "stream": True,
        "temperature": 0.1,
        "messages": messages,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            GROQ_CHAT_COMPLETIONS_URL,
            headers=headers,
            json=payload,
        ) as response:
            if response.status_code >= 400:
                error_body = await response.aread()
                logging.getLogger(__name__).error(
                    "GROQ stream_failed status=%s body=%s",
                    response.status_code,
                    error_body.decode("utf-8", errors="replace")[:2000],
                )
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                data = raw_line[6:].strip()
                if data == "[DONE]":
                    break
                parsed = json.loads(data)
                delta = parsed["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta

