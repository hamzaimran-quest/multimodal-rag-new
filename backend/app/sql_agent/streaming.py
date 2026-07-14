"""Stream LangChain SQL agent output as text tokens."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from app.security.sql_tool_wrapper import SqlAuditContext
from app.sql_agent.agent import (
    _tokenize_answer,
    build_sql_agent_executor,
    clean_sql_answer_text,
    extract_stream_text,
    finalize_sql_agent_answer,
    message_requests_tools,
    summarize_intermediate_steps,
)
from app.sql_agent.models import SqlAgentResult, SqlToolStatus

logger = logging.getLogger(__name__)

_TOOL_LABELS: dict[str, str] = {
    "sql_db_query_checker": "Checking SQL query",
    "sql_db_query": "Running database query",
}


def _tool_status(name: str, status: str) -> SqlToolStatus:
    label = _TOOL_LABELS.get(name, "Querying database")
    return SqlToolStatus(name=name, status=status, label=label)  # type: ignore[arg-type]


def _preview(value: Any, *, limit: int = 500) -> str:
    text = str(value if value is not None else "")
    text = " ".join(text.split())
    return text[:limit]


async def _yield_answer_tokens(text: str) -> AsyncIterator[str]:
    cleaned = clean_sql_answer_text(text)
    for token in _tokenize_answer(cleaned):
        yield token


async def stream_sql_agent(
    *,
    connection_url: str,
    description: str,
    question: str,
    schema_digest: str | None = None,
    audit: SqlAuditContext | None = None,
) -> AsyncIterator[str | SqlToolStatus | SqlAgentResult]:
    """Yield tool placeholders, final-answer tokens, then SqlAgentResult."""
    logger.info(
        "SQL_AGENT start question=%r schema_digest_chars=%s connection=%s",
        question[:200],
        len(schema_digest or ""),
        (audit.connection_label if audit else None),
    )
    executor = build_sql_agent_executor(
        connection_url=connection_url,
        description=description,
        schema_digest=schema_digest,
        audit=audit,
    )
    intermediate_steps: list = []
    final_output = ""
    answer_parts: list[str] = []
    generation_parts: list[str] = []
    streaming_final = False
    streamed_answer = False
    tool_calls_seen: list[str] = []
    used_invoke_fallback = False

    try:
        async for event in executor.astream_events({"input": question}, version="v2"):
            kind = event.get("event")

            if kind == "on_tool_start":
                streaming_final = False
                generation_parts.clear()
                tool_name = str(event.get("name") or "unknown_tool")
                tool_calls_seen.append(tool_name)
                tool_input = (event.get("data") or {}).get("input")
                logger.info(
                    "SQL_AGENT tool_start name=%s input_preview=%r",
                    tool_name,
                    _preview(tool_input),
                )
                yield _tool_status(tool_name, "running")

            elif kind == "on_tool_end":
                tool_name = str(event.get("name") or "unknown_tool")
                tool_output = (event.get("data") or {}).get("output")
                output_text = _preview(tool_output, limit=800)
                logger.info(
                    "SQL_AGENT tool_end name=%s output_chars=%s output_preview=%r",
                    tool_name,
                    len(str(tool_output or "")),
                    output_text,
                )
                yield _tool_status(tool_name, "complete")
                if tool_name == "sql_db_query":
                    streaming_final = True
                    generation_parts.clear()

            elif kind == "on_chat_model_stream":
                text = extract_stream_text(event)
                if not text:
                    continue
                generation_parts.append(text)
                if streaming_final:
                    answer_parts.append(text)
                    streamed_answer = True
                    yield text

            elif kind == "on_chat_model_end":
                message = event.get("data", {}).get("output")
                if message_requests_tools(message):
                    generation_parts.clear()
                    continue
                if streaming_final:
                    generation_parts.clear()
                    continue
                raw_segment = "".join(generation_parts)
                segment = clean_sql_answer_text(raw_segment)
                generation_parts.clear()
                if not segment:
                    if raw_segment.strip():
                        logger.warning(
                            "SQL_AGENT chat_segment_empty_after_clean raw_preview=%r",
                            _preview(raw_segment),
                        )
                    continue
                streamed_answer = True
                answer_parts.append(segment)
                async for token in _yield_answer_tokens(segment):
                    yield token

            elif kind == "on_chain_end" and event.get("name") == "AgentExecutor":
                output = event.get("data", {}).get("output") or {}
                if isinstance(output, dict):
                    if output.get("intermediate_steps"):
                        intermediate_steps = output.get("intermediate_steps") or []
                    if output.get("output"):
                        final_output = str(output.get("output"))
                logger.info(
                    "SQL_AGENT chain_end final_output_chars=%s final_output_preview=%r steps=%s",
                    len(final_output),
                    _preview(final_output),
                    summarize_intermediate_steps(intermediate_steps),
                )

    except Exception:
        logger.warning("SQL agent astream_events failed; falling back to invoke", exc_info=True)
        used_invoke_fallback = True
        response = executor.invoke({"input": question})
        final_output = str(response.get("output") or "")
        intermediate_steps = response.get("intermediate_steps") or []
        streamed_answer = False
        logger.info(
            "SQL_AGENT invoke_fallback final_output_chars=%s final_output_preview=%r steps=%s",
            len(final_output),
            _preview(final_output),
            summarize_intermediate_steps(intermediate_steps),
        )

    if not streamed_answer:
        fallback = clean_sql_answer_text(final_output)
        logger.info(
            "SQL_AGENT using_final_output_fallback streamed_answer=%s cleaned_chars=%s raw_chars=%s",
            streamed_answer,
            len(fallback),
            len(final_output),
        )
        answer_parts = [fallback] if fallback else []
        async for token in _yield_answer_tokens(fallback):
            yield token

    joined_parts = "".join(answer_parts)
    prose = joined_parts or final_output
    final_text, queries = finalize_sql_agent_answer(
        prose_answer=prose,
        steps=intermediate_steps,
        tool_calls_seen=tool_calls_seen,
    )
    logger.info(
        "SQL_AGENT done tools=%s streamed_answer=%s invoke_fallback=%s "
        "answer_chars=%s queries=%s joined_parts_chars=%s final_output_chars=%s "
        "answer_preview=%r",
        tool_calls_seen,
        streamed_answer,
        used_invoke_fallback,
        len(final_text),
        queries,
        len(joined_parts),
        len(final_output),
        _preview(final_text),
    )
    if not queries and "sql_db_query" not in tool_calls_seen:
        logger.warning(
            "SQL_AGENT missing_execute question=%r tools=%s steps=%s final_output_preview=%r",
            question[:200],
            tool_calls_seen,
            summarize_intermediate_steps(intermediate_steps),
            _preview(final_output),
        )
    yield SqlAgentResult(
        answer_text=final_text,
        queries=queries,
        route_mode="sql",
    )
