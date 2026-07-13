"""Stream LangChain SQL agent output as text tokens."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from app.sql_agent.agent import (
    _extract_queries_from_steps,
    _tokenize_answer,
    build_sql_agent_executor,
    clean_sql_answer_text,
    extract_stream_text,
    message_requests_tools,
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
) -> AsyncIterator[str | SqlToolStatus | SqlAgentResult]:
    """Yield tool placeholders, final-answer tokens, then SqlAgentResult."""
    executor = build_sql_agent_executor(
        connection_url=connection_url,
        description=description,
        schema_digest=schema_digest,
    )
    intermediate_steps: list = []
    final_output = ""
    answer_parts: list[str] = []
    generation_parts: list[str] = []
    streaming_final = False
    streamed_answer = False

    try:
        async for event in executor.astream_events({"input": question}, version="v2"):
            kind = event.get("event")

            if kind == "on_tool_start":
                streaming_final = False
                generation_parts.clear()
                tool_name = str(event.get("name") or "query_database")
                yield _tool_status(tool_name, "running")

            elif kind == "on_tool_end":
                tool_name = str(event.get("name") or "query_database")
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
                segment = clean_sql_answer_text("".join(generation_parts))
                generation_parts.clear()
                if not segment:
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

    except Exception:
        logger.warning("SQL agent astream_events failed; falling back to invoke", exc_info=True)
        response = executor.invoke({"input": question})
        final_output = str(response.get("output") or "")
        intermediate_steps = response.get("intermediate_steps") or []
        streamed_answer = False

    if not streamed_answer:
        fallback = clean_sql_answer_text(final_output)
        answer_parts = [fallback] if fallback else []
        async for token in _yield_answer_tokens(fallback):
            yield token

    final_text = clean_sql_answer_text("".join(answer_parts)) or clean_sql_answer_text(final_output)
    yield SqlAgentResult(
        answer_text=final_text,
        queries=_extract_queries_from_steps(intermediate_steps),
        route_mode="sql",
    )
