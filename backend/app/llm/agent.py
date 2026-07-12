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
from app.charts.auto import chart_requested
from app.llm.query_rewrite import rewrite_query_for_retrieval
from app.retrieval.scope import scope_hint_for_agent

logger = logging.getLogger(__name__)

AGENT_ROUTER_PROMPT = """You are the routing assistant for a document Q&A product.

Decide how to handle each user message across one or more tool rounds:

- **Greetings, thanks, small talk, or questions about what you can do** — reply directly in plain text. Do **not** call any tools.
- **Ambiguous factual questions** (unclear what the user wants, missing key details) — ask a short clarifying question in plain text. Do **not** ask which document to use; document scope is set in the UI. **Exception:** when document scope is restricted in the UI, treat the question as being about the selected file(s) and call `search_documents` instead of asking generic clarification.
- **Questions about content inside uploaded documents** — call `search_documents` with a **standalone** search query.
- **User explicitly asks to see a photo, portrait, or figure** — call `search_images` (optionally also `search_documents` for text context). Do **not** use `search_images` for data charts or graphs.
- **User asks to create, draw, plot, or visualize a chart/graph from document data** — call **`create_chart` only** with a standalone query. `create_chart` finds and charts the table internally; do **not** call `search_documents` or `search_images` instead.
- **User asks what files are in scope or what they have uploaded** — call `list_documents`.

## Multi-step routing

- You may call tools across multiple rounds. After tool results, either search again with a refined/broader query, ask clarification about the question (not scope), or **stop** (the system answers from retrieved chunks — do not summarize search results yourself).
- If the first search returns weak or zero hits, try `search_documents` again with different keywords or broader terms.
- Stop calling tools once retrieval is sufficient; never invent document facts in the routing turn.

Rules:
- Never answer document factual questions in plain text — always call `search_documents`.
- Document scope is configured in the UI; never choose, filter, or ask about which document to search.
- Keep direct replies brief — only for greetings, thanks, meta help, or genuine clarification questions about the user's intent.
- Prefer `top_k` of 5 for `search_documents` (do not exceed 5).
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
                        "description": "Number of chunks to retrieve (1-5). Omit to use the default of 5.",
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
        "description": "Retrieve image chunks when the user explicitly wants to see a photo, portrait, or figure.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Standalone visual search query (resolve pronouns from chat history).",
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
            "Build a bar or line chart from document table data (QuickChart image). "
            "Call this when the user asks to create, draw, plot, or visualize a chart or graph. "
            "Searches for a relevant table internally — do not call search_documents first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Standalone search query to find the relevant table when chunk_id is omitted.",
                },
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line"],
                    "description": "Requested chart type. Omit to use the best fit for the table shape.",
                },
                "chunk_id": {
                    "type": "string",
                    "description": "Table chunk id from search_documents when already known.",
                },
            },
            "required": ["query"],
        },
    },
}

AGENT_TOOLS: list[dict[str, Any]] = [*PHASE_A_TOOLS, SEARCH_IMAGES_TOOL, CREATE_CHART_TOOL]


def _chart_routing_hint(user_query: str) -> str:
    if not chart_requested(user_query):
        return ""
    return (
        "\n\nChart request detected: call `create_chart` with a standalone query. "
        "Do not call `search_documents` or `search_images` for this turn."
    )


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


def _is_clarification_reply(content: str) -> bool:
    """Router asked the user to disambiguate the question — not document scope."""
    text = content.strip()
    if not text or "?" not in text:
        return False
    lower = text.lower()
    scope_markers = (
        "which document",
        "which file",
        "uploaded documents",
        "what document",
        "what file",
    )
    if any(marker in lower for marker in scope_markers):
        return False
    markers = (
        "could you clarify",
        "please specify",
        "do you mean",
        "which one",
        "can you tell me which",
        "more specific",
        "clarify",
    )
    return any(marker in lower for marker in markers)


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


def _force_search_documents(
    client: Any,
    *,
    user_id: int,
    search_query: str,
    scope_doc_ids: list[str] | None,
    default_top_k: int,
    anchor_fallback_query: str | None = None,
) -> tuple[list[RetrievedChunk], str]:
    """Fallback when the router wrongly skips tools for a document question."""
    result_json, chunks, _, _ = _execute_tool_call(
        client,
        user_id=user_id,
        name="search_documents",
        arguments_json=json.dumps({"query": search_query, "top_k": default_top_k}),
        scope_doc_ids=scope_doc_ids,
        default_top_k=default_top_k,
        anchor_fallback_query=anchor_fallback_query or search_query,
    )
    return chunks, result_json


def _execute_tool_call(
    client: Any,
    *,
    user_id: int,
    name: str,
    arguments_json: str,
    scope_doc_ids: list[str] | None,
    default_top_k: int,
    prior_table_chunk_ids: list[str] | None = None,
    anchor_fallback_query: str | None = None,
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
        payload, chunks = execute_search_documents(
            client,
            user_id=user_id,
            query=query,
            top_k=top_k,
            scope_doc_ids=scope_doc_ids,
            anchor_fallback_query=anchor_fallback_query,
        )
        return payload, chunks, [], []

    if name == "list_documents":
        return execute_list_documents(client, user_id=user_id, scope_doc_ids=scope_doc_ids), [], [], []

    if name == "search_images":
        query = str(args.get("query", "")).strip()
        if not query:
            return json.dumps({"error": "query is required"}), [], [], []
        payload, images = execute_search_images(
            client,
            user_id=user_id,
            query=query,
            scope_doc_ids=scope_doc_ids,
        )
        return payload, [], images, []

    if name == "create_chart":
        query = str(args.get("query", "")).strip()
        if not query:
            return json.dumps({"error": "query is required"}), [], [], []
        chart_type = args.get("chart_type")
        chunk_id = args.get("chunk_id")
        payload, charts, source_chunks = execute_create_chart(
            client,
            user_id=user_id,
            query=query,
            chart_type=str(chart_type).strip().lower() if chart_type else None,
            scope_doc_ids=scope_doc_ids,
            chunk_id=str(chunk_id).strip() if chunk_id else None,
            prior_table_chunk_ids=prior_table_chunk_ids,
            top_k=default_top_k,
        )
        return payload, source_chunks, [], charts

    return json.dumps({"error": f"unknown_tool:{name}"}), [], [], []


async def run_agent_turn(
    client: Any,
    *,
    user_id: int,
    user_query: str,
    prior_queries: list[str] | None = None,
    last_assistant_reply: str | None = None,
    prior_table_chunk_ids: list[str] | None = None,
    scope_doc_ids: list[str] | None = None,
    scoped_filenames: list[str] | None = None,
    default_top_k: int | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> AgentTurnResult:
    """Router LLM loop; executes tools across rounds; prepares for grounded answer stream."""
    result: AgentTurnResult | None = None
    async for event in iter_agent_turn(
        client,
        user_id=user_id,
        user_query=user_query,
        prior_queries=prior_queries,
        last_assistant_reply=last_assistant_reply,
        prior_table_chunk_ids=prior_table_chunk_ids,
        scope_doc_ids=scope_doc_ids,
        scoped_filenames=scoped_filenames,
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
    prior_queries: list[str] | None = None,
    last_assistant_reply: str | None = None,
    prior_table_chunk_ids: list[str] | None = None,
    scope_doc_ids: list[str] | None = None,
    scoped_filenames: list[str] | None = None,
    default_top_k: int | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Multi-round agent loop; yields tool events then a final complete event."""
    top_k = default_top_k or settings.default_top_k
    tool_defs = tools if tools is not None else AGENT_TOOLS
    max_rounds = settings.agent_max_rounds

    router_query = await rewrite_query_for_retrieval(
        user_query,
        prior_queries,
        last_assistant_reply=last_assistant_reply,
    )

    logger.info(
        "AGENT turn_start user_id=%s prior_queries=%s query_preview=%r router_query_preview=%r scope_doc_ids=%s max_rounds=%s",
        user_id,
        len(prior_queries or []),
        user_query[:120],
        router_query[:120],
        scope_doc_ids,
        max_rounds,
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": AGENT_ROUTER_PROMPT
            + scope_hint_for_agent(scope_doc_ids, scoped_filenames=scoped_filenames)
            + _chart_routing_hint(user_query),
        },
        {"role": "user", "content": router_query},
    ]

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

                if content and _is_clarification_reply(content) and not scope_doc_ids:
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

                if content and _is_clarification_reply(content) and scope_doc_ids:
                    logger.info(
                        "AGENT route_scoped_clarification_fallback query_preview=%r scope_count=%s",
                        user_query[:120],
                        len(scope_doc_ids),
                    )

                logger.warning(
                    "AGENT route_fallback query_preview=%r reason=%s",
                    user_query[:120],
                    "router_skipped_tools"
                    if not (content and _is_clarification_reply(content))
                    else "scoped_clarification_blocked",
                )
                search_query = router_query
                chunks, _ = _force_search_documents(
                    client,
                    user_id=user_id,
                    search_query=search_query,
                    scope_doc_ids=scope_doc_ids,
                    default_top_k=top_k,
                    anchor_fallback_query=user_query,
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
                scope_doc_ids=scope_doc_ids,
                default_top_k=top_k,
                prior_table_chunk_ids=prior_table_chunk_ids,
                anchor_fallback_query=user_query,
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
