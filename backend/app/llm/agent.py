"""Groq tool-calling agent: multi-round routing, retrieval, and grounded answers."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import settings
from app.llm.groq import GROQ_CHAT_COMPLETIONS_URL, stream_groq_messages
from app.llm.tools import (
    execute_create_chart,
    execute_list_documents,
    execute_search_documents,
    execute_search_images,
)
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

FOLLOWUP_MARKERS = re.compile(
    r"\b(him|her|them|they|it|that|this|those|these|same|above|earlier|previous|there)\b",
    re.IGNORECASE,
)

AGENT_ROUTER_PROMPT = """You are the routing assistant for a document Q&A product.

Decide how to handle each user message across one or more tool rounds:

- **Greetings, thanks, small talk, or questions about what you can do** — reply directly in plain text. Do **not** call any tools.
- **Truly ambiguous scope** (multiple documents, unclear referent) — ask a short clarifying question in plain text. Do **not** call tools yet.
- **Questions about content inside uploaded documents** — call `search_documents` with a **standalone** search query.
- **User explicitly asks to see a photo, portrait, figure, or chart** — call `search_images` (optionally also `search_documents` for text context).
- **User asks to create, draw, plot, or visualize a chart/graph from document data** — call `search_documents` first if needed, then `create_chart` with the requested chart type (`bar`, `line`, or `pie`). If chart creation fails, explain that the data is not chartable.
- **User asks what files they have, or which document to use** — call `list_documents`.

## Follow-ups and chat history

- Rewrite follow-ups into standalone queries using prior turns (resolve pronouns like him/her/it/that to the actual person or topic).
- Example: prior turn discussed "Ren Zhengfei" → "show an image of him" becomes search_images query "Ren Zhengfei portrait photo".

## Multi-step routing

- You may call tools across multiple rounds. After tool results, either search again with a refined/broader query, ask clarification, or **stop** (the system answers from retrieved chunks — do not summarize search results yourself).
- If the first search returns weak or zero hits, try `search_documents` again with different keywords or broader terms.
- When multiple documents exist and the user did not name one, `list_documents` first, then `search_documents` with `doc_id` if needed.
- Stop calling tools once retrieval is sufficient; never invent document facts in the routing turn.

Rules:
- Never answer document factual questions in plain text — always call `search_documents` (or `list_documents` then `search_documents`).
- Keep direct replies brief — only for greetings, thanks, meta help, or genuine clarification questions.
- When calling `search_documents`, pass `doc_id` only when scope clearly targets one file.
- Prefer `top_k` of 8–12 for factual questions unless the user asks for a narrow snippet.
- Use the native tool-calling API only. Never write `<function=...>` tags or other custom function syntax."""

PHASE_A_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Hybrid search over the user's indexed document chunks (text, tables, images metadata).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Standalone search query for retrieval.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve (1-50).",
                    },
                    "doc_id": {
                        "type": "string",
                        "description": "Optional document scope filter.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List documents the user has uploaded with status and chunk counts.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]

SEARCH_IMAGES_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_images",
        "description": "Retrieve image chunks when the user explicitly wants to see a photo, portrait, figure, or chart.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Standalone visual search query (resolve pronouns from chat history).",
                },
                "doc_id": {
                    "type": "string",
                    "description": "Optional document scope filter.",
                },
            },
            "required": ["query"],
        },
    },
}

CREATE_CHART_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "create_chart",
        "description": (
            "Build a bar, line, or pie chart from document table data. "
            "Use when the user asks to create, plot, or visualize a chart. "
            "Fails closed when the retrieved table is not structurally chartable."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Standalone search query to find the relevant table.",
                },
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "pie"],
                    "description": "Requested chart type. Omit to use the best fit for the table shape.",
                },
                "doc_id": {
                    "type": "string",
                    "description": "Optional document scope filter.",
                },
                "chunk_id": {
                    "type": "string",
                    "description": "Optional specific table chunk id when already known from search.",
                },
                "period_label": {
                    "type": "string",
                    "description": "For pie charts from multi-period tables, which period column to slice (e.g. 2024).",
                },
            },
            "required": ["query"],
        },
    },
}

AGENT_TOOLS: list[dict[str, Any]] = [*PHASE_A_TOOLS, SEARCH_IMAGES_TOOL, CREATE_CHART_TOOL]

REWRITE_PROMPT = """Rewrite the user's message into a standalone search query for document retrieval.

Use the chat history to resolve pronouns and vague references (him, her, it, that, they, the same, above).
Output ONLY the rewritten query — no quotes, explanation, or punctuation beyond what the query needs."""


@dataclass
class AgentTurnResult:
    """Outcome of one agent turn before answer streaming."""

    direct_answer: str | None = None
    is_clarification: bool = False
    answer_messages: list[dict[str, Any]] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    intent_images: list[dict[str, Any]] = field(default_factory=list)
    tool_charts: list[dict[str, Any]] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    rounds_used: int = 0


def _scope_hint(default_doc_id: str | None) -> str:
    if not default_doc_id:
        return ""
    return f"\n\nSession scope: restrict search to doc_id={default_doc_id!r} unless the user names another file."


def _recover_tool_calls_from_failed_generation(failed_generation: str) -> list[dict[str, Any]] | None:
    """
    Groq sometimes rejects model output that used <function=name{...}</function> instead
    of native tool_calls. Parse that text and rebuild OpenAI-style tool_calls.
    """
    text = failed_generation.strip()
    if "<function=" not in text:
        return None

    recovered: list[dict[str, Any]] = []
    for fragment in re.findall(r"<function=[^<]+(?:</function>)?", text):
        body = fragment.removeprefix("<function=").removesuffix("</function>").strip()
        brace_idx = body.find("{")
        if brace_idx < 0:
            continue
        name = body[:brace_idx].strip()
        if not name:
            continue
        try:
            args, _end = json.JSONDecoder().raw_decode(body[brace_idx:])
        except json.JSONDecodeError:
            continue
        if not isinstance(args, dict):
            continue
        recovered.append(
            {
                "id": f"call_recovered_{len(recovered)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        )

    return recovered or None


async def groq_chat_completion(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    model: str | None = None,
    temperature: float = 0.1,
) -> dict[str, Any]:
    if not settings.groq_configured:
        raise RuntimeError("GROQ_API_KEY is not configured")

    chosen_model = model or settings.agent_model
    payload: dict[str, Any] = {
        "model": chosen_model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        payload["parallel_tool_calls"] = False

    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }
    logger.info(
        "AGENT groq_request model=%s message_count=%s tools=%s",
        chosen_model,
        len(messages),
        len(tools or []),
    )
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(GROQ_CHAT_COMPLETIONS_URL, headers=headers, json=payload)
        if response.status_code >= 400:
            if response.status_code == 400 and tools:
                try:
                    error_body = response.json()
                    error = error_body.get("error") or {}
                    if error.get("code") == "tool_use_failed":
                        failed_generation = str(error.get("failed_generation") or "")
                        tool_calls = _recover_tool_calls_from_failed_generation(failed_generation)
                        if tool_calls:
                            logger.warning(
                                "AGENT groq_tool_use_failed_recovered tools=%s preview=%r",
                                [
                                    (tc.get("function") or {}).get("name")
                                    for tc in tool_calls
                                ],
                                failed_generation[:200],
                            )
                            return {
                                "choices": [
                                    {
                                        "message": {
                                            "content": None,
                                            "tool_calls": tool_calls,
                                        },
                                        "finish_reason": "tool_calls",
                                    }
                                ]
                            }
                except Exception:
                    logger.warning(
                        "AGENT groq_tool_use_failed_parse_error body=%s",
                        response.text[:500],
                        exc_info=True,
                    )
            logger.error(
                "AGENT groq_request_failed status=%s model=%s body=%s",
                response.status_code,
                chosen_model,
                response.text[:2000],
            )
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls") or []
        logger.info(
            "AGENT groq_response finish_reason=%s tool_calls=%s direct_chars=%s",
            choice.get("finish_reason"),
            [((tc.get("function") or {}).get("name")) for tc in tool_calls],
            len((message.get("content") or "")),
        )
        return body


def _is_small_talk(query: str) -> bool:
    """True only for greetings/meta — not for document factual questions."""
    normalized = query.strip().lower().rstrip("?!.")
    if not normalized:
        return True
    exact = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
        "ok",
        "okay",
        "good morning",
        "good afternoon",
        "good evening",
    }
    if normalized in exact:
        return True
    if normalized.startswith(("hi ", "hello ", "hey ")) and len(normalized) < 40:
        return True
    if "what can you do" in normalized or "how can you help" in normalized:
        return True
    return False


def _is_clarification_reply(content: str) -> bool:
    """Router asked the user to disambiguate — valid direct reply without search."""
    text = content.strip()
    if not text or "?" not in text:
        return False
    lower = text.lower()
    markers = (
        "which document",
        "which file",
        "could you clarify",
        "please specify",
        "do you mean",
        "which one",
        "can you tell me which",
        "uploaded documents",
        "more specific",
        "clarify",
    )
    return any(marker in lower for marker in markers)


def _needs_followup_rewrite(query: str, history: list[dict[str, str]]) -> bool:
    return bool(history) and bool(FOLLOWUP_MARKERS.search(query))


def _merge_retrieved_chunks(
    existing: list[RetrievedChunk],
    new_chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    by_id = {chunk.chunk_id: chunk for chunk in existing}
    for chunk in new_chunks:
        prior = by_id.get(chunk.chunk_id)
        if prior is None or chunk.score > prior.score:
            by_id[chunk.chunk_id] = chunk
    return sorted(by_id.values(), key=lambda c: c.score, reverse=True)


async def rewrite_retrieval_query(
    user_query: str,
    history: list[dict[str, str]],
) -> str:
    """Resolve pronouns/vague follow-ups into a standalone retrieval query."""
    if not _needs_followup_rewrite(user_query, history):
        return user_query

    messages: list[dict[str, Any]] = [{"role": "system", "content": REWRITE_PROMPT}]
    messages.extend(history[-(settings.agent_history_turns * 2) :])
    messages.append({"role": "user", "content": user_query})

    try:
        completion = await groq_chat_completion(messages=messages, tools=None, temperature=0.0)
        rewritten = (completion["choices"][0]["message"].get("content") or "").strip()
        rewritten = rewritten.strip("\"'")
        if rewritten and len(rewritten) <= 500 and rewritten.lower() != user_query.strip().lower():
            logger.info("AGENT query_rewrite original=%r rewritten=%r", user_query[:80], rewritten[:80])
            return rewritten
    except Exception:
        logger.warning("AGENT query_rewrite_failed query=%r", user_query[:80], exc_info=True)
    return user_query


def _force_search_documents(
    client: Any,
    *,
    user_id: int,
    search_query: str,
    default_doc_id: str | None,
    default_top_k: int,
) -> tuple[list[RetrievedChunk], str]:
    """Fallback when the router wrongly skips tools for a document question."""
    result_json, chunks, _, _ = _execute_tool_call(
        client,
        user_id=user_id,
        name="search_documents",
        arguments_json=json.dumps({"query": search_query, "top_k": default_top_k}),
        default_doc_id=default_doc_id,
        default_top_k=default_top_k,
    )
    return chunks, result_json


def _sanitize_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """Trim history so router requests stay within Groq limits."""
    if not history:
        return []
    max_chars = settings.agent_history_max_chars
    cleaned: list[dict[str, str]] = []
    for item in history:
        role = item.get("role")
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        if len(content) > max_chars:
            content = content[: max_chars - 1].rstrip() + "…"
        cleaned.append({"role": role, "content": content})
    return cleaned


def _execute_tool_call(
    client: Any,
    *,
    user_id: int,
    name: str,
    arguments_json: str,
    default_doc_id: str | None,
    default_top_k: int,
) -> tuple[str, list[RetrievedChunk], list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError:
        args = {}

    if name == "search_documents":
        query = str(args.get("query", "")).strip()
        if not query:
            return json.dumps({"error": "query is required"}), [], [], []
        top_k = args.get("top_k", default_top_k)
        doc_id = args.get("doc_id")
        payload, chunks = execute_search_documents(
            client,
            user_id=user_id,
            query=query,
            top_k=top_k,
            doc_id=doc_id,
            default_doc_id=default_doc_id,
        )
        return payload, chunks, [], []

    if name == "list_documents":
        return execute_list_documents(client, user_id=user_id), [], [], []

    if name == "search_images":
        query = str(args.get("query", "")).strip()
        if not query:
            return json.dumps({"error": "query is required"}), [], [], []
        doc_id = args.get("doc_id")
        payload, images = execute_search_images(
            client,
            user_id=user_id,
            query=query,
            doc_id=doc_id,
            default_doc_id=default_doc_id,
        )
        return payload, [], images, []

    if name == "create_chart":
        query = str(args.get("query", "")).strip()
        if not query:
            return json.dumps({"error": "query is required"}), [], [], []
        chart_type = args.get("chart_type")
        doc_id = args.get("doc_id")
        chunk_id = args.get("chunk_id")
        period_label = args.get("period_label")
        payload, charts = execute_create_chart(
            client,
            user_id=user_id,
            query=query,
            chart_type=str(chart_type).strip().lower() if chart_type else None,
            doc_id=doc_id,
            default_doc_id=default_doc_id,
            chunk_id=str(chunk_id).strip() if chunk_id else None,
            period_label=str(period_label).strip() if period_label else None,
            top_k=default_top_k,
        )
        return payload, [], [], charts

    return json.dumps({"error": f"unknown_tool:{name}"}), [], [], []


async def run_agent_turn(
    client: Any,
    *,
    user_id: int,
    user_query: str,
    history: list[dict[str, str]] | None = None,
    default_doc_id: str | None = None,
    default_top_k: int | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> AgentTurnResult:
    """Router LLM loop; executes tools across rounds; prepares for grounded answer stream."""
    result: AgentTurnResult | None = None
    async for event in iter_agent_turn(
        client,
        user_id=user_id,
        user_query=user_query,
        history=history,
        default_doc_id=default_doc_id,
        default_top_k=default_top_k,
        tools=tools,
    ):
        if event["type"] == "complete":
            result = event["result"]
    assert result is not None
    return result


async def iter_agent_turn(
    client: Any,
    *,
    user_id: int,
    user_query: str,
    history: list[dict[str, str]] | None = None,
    default_doc_id: str | None = None,
    default_top_k: int | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Multi-round agent loop; yields tool events then a final complete event."""
    top_k = default_top_k or settings.default_top_k
    tool_defs = tools if tools is not None else AGENT_TOOLS
    history = _sanitize_history(history)
    max_rounds = settings.agent_max_rounds

    logger.info(
        "AGENT turn_start user_id=%s history_turns=%s query_preview=%r scoped_doc=%s max_rounds=%s",
        user_id,
        len(history) // 2,
        user_query[:120],
        default_doc_id,
        max_rounds,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_ROUTER_PROMPT + _scope_hint(default_doc_id)},
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_query})

    retrieved: list[RetrievedChunk] = []
    intent_images: list[dict[str, Any]] = []
    tool_charts: list[dict[str, Any]] = []
    tools_used: list[str] = []
    rounds_used = 0

    for round_num in range(1, max_rounds + 1):
        rounds_used = round_num
        completion = await groq_chat_completion(messages=messages, tools=tool_defs)
        message = completion["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            content = (message.get("content") or "").strip()

            if not tools_used:
                if _is_small_talk(user_query):
                    logger.info("AGENT route_direct_reply chars=%s", len(content))
                    yield {
                        "type": "complete",
                        "result": AgentTurnResult(
                            direct_answer=content or "Hello! How can I help with your documents?",
                            rounds_used=rounds_used,
                        ),
                    }
                    return

                if content and _is_clarification_reply(content):
                    logger.info("AGENT route_clarification chars=%s", len(content))
                    yield {
                        "type": "complete",
                        "result": AgentTurnResult(
                            direct_answer=content,
                            is_clarification=True,
                            rounds_used=rounds_used,
                        ),
                    }
                    return

                logger.warning(
                    "AGENT route_fallback query_preview=%r reason=router_skipped_tools",
                    user_query[:120],
                )
                search_query = await rewrite_retrieval_query(user_query, history)
                chunks, _ = _force_search_documents(
                    client,
                    user_id=user_id,
                    search_query=search_query,
                    default_doc_id=default_doc_id,
                    default_top_k=top_k,
                )
                yield {
                    "type": "complete",
                    "result": AgentTurnResult(
                        retrieved_chunks=chunks,
                        tools_used=["search_documents"],
                        rounds_used=rounds_used,
                    ),
                }
                return

            logger.info(
                "AGENT route_stop_after_tools rounds=%s tools=%s chunks=%s",
                round_num,
                tools_used,
                len(retrieved),
            )
            break

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": message.get("content"),
            "tool_calls": tool_calls,
        }
        messages.append(assistant_message)

        for call in tool_calls:
            fn = call.get("function") or {}
            name = fn.get("name", "")
            call_id = call.get("id") or f"call_{round_num}_{name}"
            tools_used.append(name)

            yield {"type": "tool", "name": name, "status": "running", "round": round_num}
            logger.info("AGENT tool_start round=%s name=%s args=%s", round_num, name, fn.get("arguments", "{}")[:500])

            result_json, chunks, images, charts = _execute_tool_call(
                client,
                user_id=user_id,
                name=name,
                arguments_json=fn.get("arguments", "{}"),
                default_doc_id=default_doc_id,
                default_top_k=top_k,
            )
            retrieved = _merge_retrieved_chunks(retrieved, chunks)
            intent_images.extend(images)
            if charts:
                tool_charts.extend(charts)

            logger.info(
                "AGENT tool_done round=%s name=%s chunks=%s images=%s charts=%s result_chars=%s total_chunks=%s",
                round_num,
                name,
                len(chunks),
                len(images),
                len(charts),
                len(result_json),
                len(retrieved),
            )
            yield {"type": "tool", "name": name, "status": "complete", "round": round_num}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result_json,
                }
            )

        logger.info(
            "AGENT round_complete round=%s tools=%s retrieved_chunks=%s intent_images=%s",
            round_num,
            tools_used,
            len(retrieved),
            len(intent_images),
        )

    logger.info(
        "AGENT turn_ready rounds=%s tools=%s retrieved_chunks=%s intent_images=%s",
        rounds_used,
        tools_used,
        len(retrieved),
        len(intent_images),
    )
    yield {
        "type": "complete",
        "result": AgentTurnResult(
            retrieved_chunks=retrieved,
            intent_images=intent_images,
            tool_charts=tool_charts,
            tools_used=tools_used,
            rounds_used=rounds_used,
        ),
    }


async def stream_agent_answer(messages: list[dict[str, Any]]):
    """Stream the final grounded answer from prepared agent messages."""
    async for token in stream_groq_messages(messages=messages):
        yield token
