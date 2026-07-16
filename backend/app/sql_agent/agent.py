"""LangChain SQL agent construction and execution."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.config import settings
from app.security.sql_tool_wrapper import SqlAuditContext, secure_sql_tools
from app.sql_agent.llm import build_sql_agent_llm
from app.sql_agent.models import SqlAgentResult
from app.sql_agent.prompts import build_sql_agent_prefix

logger = logging.getLogger(__name__)

_SQL_QUERY_RE = re.compile(r"SELECT\b", re.IGNORECASE)
_SQL_LEADING_RE = re.compile(r"^\s*SELECT\b.+?;\s*", re.IGNORECASE | re.DOTALL)
_SQL_CODE_BLOCK_RE = re.compile(r"```(?:sql)?\s*.*?```\s*", re.IGNORECASE | re.DOTALL)

_NO_EXECUTE_ANSWER = "Not found in the provided database results."


def _is_sql_execute_tool(tool: Any) -> bool:
    name = str(getattr(tool, "tool", None) or tool or "")
    return name == "sql_db_query"


def sql_agent_tool_names() -> set[str]:
    """LangChain SQL tools exposed to the agent for this deployment."""
    names = {"sql_db_query"}
    if settings.sql_agent_query_checker_enabled:
        names.add("sql_db_query_checker")
    return names


def _filter_sql_agent_tools(tools: list[Any]) -> list[Any]:
    allowed = sql_agent_tool_names()
    return [tool for tool in tools if getattr(tool, "name", "") in allowed]


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().rstrip(";")).lower()


def clean_sql_answer_text(text: str) -> str:
    """Remove SQL statements or code blocks accidentally echoed in the user-facing answer."""
    cleaned = (text or "").strip()
    before_chars = len(cleaned)
    cleaned = _SQL_LEADING_RE.sub("", cleaned, count=1)
    cleaned = _SQL_CODE_BLOCK_RE.sub("", cleaned)
    cleaned = cleaned.strip()
    if before_chars and not cleaned:
        logger.warning(
            "SQL_CLEAN wiped_answer before_chars=%s preview=%r",
            before_chars,
            (text or "")[:300],
        )
    elif before_chars != len(cleaned):
        logger.info(
            "SQL_CLEAN trimmed before_chars=%s after_chars=%s",
            before_chars,
            len(cleaned),
        )
    return cleaned


def summarize_intermediate_steps(steps: list[Any]) -> list[dict[str, Any]]:
    """Compact tool call/observation summary for diagnostics."""
    summary: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        if not isinstance(step, tuple) or len(step) < 2:
            summary.append({"index": index, "shape": type(step).__name__})
            continue
        action, observation = step[0], step[1]
        tool_name = str(getattr(action, "tool", None) or "")
        tool_input = getattr(action, "tool_input", None)
        if isinstance(tool_input, dict):
            input_preview = str(
                tool_input.get("query") or tool_input.get("input") or tool_input
            )[:400]
        else:
            input_preview = str(tool_input or "")[:400]
        observation_text = str(observation or "")
        summary.append(
            {
                "index": index,
                "tool": tool_name,
                "input_preview": input_preview,
                "observation_chars": len(observation_text),
                "observation_preview": observation_text[:500],
            }
        )
    return summary


def agent_executed_sql_query(
    steps: list[Any],
    tool_calls_seen: list[str] | None = None,
) -> bool:
    """True when sql_db_query actually ran with a SELECT (not merely attempted)."""
    del tool_calls_seen  # retained for call-site logging; not sufficient alone
    if _extract_queries_from_steps(steps):
        return True
    return bool(extract_query_observations(steps))


def extract_query_observations(steps: list[Any]) -> str:
    """Join observations from successful sql_db_query steps for answer fallback."""
    parts: list[str] = []
    for step in steps:
        if not isinstance(step, tuple) or len(step) < 2:
            continue
        action, observation = step[0], step[1]
        if not _is_sql_execute_tool(getattr(action, "tool", None)):
            continue
        text = str(observation or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def finalize_sql_agent_answer(
    *,
    prose_answer: str,
    steps: list[Any],
    tool_calls_seen: list[str] | None = None,
) -> tuple[str, list[str]]:
    """
    Require a genuine sql_db_query before trusting prose.

    If execute never ran, discard invented prose. If execute ran but prose is empty,
    fall back to tool observations so the answer LLM still has facts.
    """
    queries = _extract_queries_from_steps(steps)
    cleaned = clean_sql_answer_text(prose_answer)
    executed = agent_executed_sql_query(steps, tool_calls_seen)

    if not executed:
        logger.warning(
            "SQL_AGENT require_execute blocked_answer tools=%s steps=%s prose_preview=%r",
            tool_calls_seen,
            summarize_intermediate_steps(steps),
            (cleaned or prose_answer)[:300],
        )
        return _NO_EXECUTE_ANSWER, []

    if cleaned:
        return cleaned, queries

    observations = extract_query_observations(steps)
    if observations:
        logger.info(
            "SQL_AGENT using_query_observations chars=%s queries=%s",
            len(observations),
            queries,
        )
        return observations, queries

    logger.warning(
        "SQL_AGENT executed_but_empty_result queries=%s steps=%s",
        queries,
        summarize_intermediate_steps(steps),
    )
    return _NO_EXECUTE_ANSWER, queries


def _tokenize_answer(text: str) -> list[str]:
    if not text:
        return []
    parts = text.split(" ")
    return [part if idx == 0 else f" {part}" for idx, part in enumerate(parts)]


def extract_stream_text(event: dict[str, Any]) -> str:
    chunk = event.get("data", {}).get("chunk")
    if chunk is None:
        return ""
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return ""


def message_requests_tools(message: Any) -> bool:
    if message is None:
        return False
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        return True
    additional = getattr(message, "additional_kwargs", None) or {}
    if additional.get("tool_calls"):
        return True
    content = getattr(message, "content", None)
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"tool_use", "tool_calls"}:
                return True
    return False


def _extract_queries_from_steps(steps: list[Any]) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    for step in steps:
        if not isinstance(step, tuple) or len(step) < 2:
            continue
        action, _observation = step[0], step[1]
        if not _is_sql_execute_tool(getattr(action, "tool", None)):
            continue
        tool_input = getattr(action, "tool_input", None)
        if isinstance(tool_input, dict):
            query = tool_input.get("query") or tool_input.get("input")
        else:
            query = tool_input
        if not isinstance(query, str) or not _SQL_QUERY_RE.search(query):
            continue
        normalized = _normalize_query(query)
        if normalized in seen:
            continue
        seen.add(normalized)
        queries.append(query.strip())
    return queries


def build_sql_agent_executor(
    *,
    connection_url: str,
    description: str,
    schema_digest: str | None = None,
    audit: SqlAuditContext | None = None,
):
    if not settings.sql_agent_llm_configured:
        raise RuntimeError(
            "SQL agent LLM is not configured. Set SQL_AGENT_OPENROUTER_API_KEY "
            "or GROQ_API_KEY."
        )
    if not settings.sql_agent_enabled:
        raise RuntimeError("SQL agent is disabled")

    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_community.agent_toolkits import create_sql_agent
    from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
    from langchain_community.utilities import SQLDatabase
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    statement_timeout_ms = int(settings.sql_agent_query_timeout_seconds) * 1000
    db = SQLDatabase.from_uri(
        connection_url,
        sample_rows_in_table_info=0 if schema_digest else 3,
        engine_args={
            "connect_args": {"options": f"-c statement_timeout={statement_timeout_ms}"},
        },
    )
    llm = build_sql_agent_llm()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    prefix = build_sql_agent_prefix(description, schema_digest=schema_digest)

    if schema_digest:
        tools = _filter_sql_agent_tools(toolkit.get_tools())
        tools = secure_sql_tools(tools, audit=audit)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prefix),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        agent = create_tool_calling_agent(llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            max_iterations=settings.sql_agent_max_steps,
            return_intermediate_steps=True,
        )

    executor = create_sql_agent(
        llm,
        toolkit=toolkit,
        agent_type="tool-calling",
        verbose=False,
        prefix=prefix,
        max_iterations=settings.sql_agent_max_steps,
        agent_executor_kwargs={"return_intermediate_steps": True},
    )
    executor.tools = secure_sql_tools(_filter_sql_agent_tools(list(executor.tools)), audit=audit)
    return executor


def run_sql_agent_sync(
    *,
    connection_url: str,
    description: str,
    question: str,
    schema_digest: str | None = None,
) -> SqlAgentResult:
    executor = build_sql_agent_executor(
        connection_url=connection_url,
        description=description,
        schema_digest=schema_digest,
    )
    response = executor.invoke({"input": question})
    steps = response.get("intermediate_steps") or []
    answer, queries = finalize_sql_agent_answer(
        prose_answer=str(response.get("output") or ""),
        steps=steps,
    )
    return SqlAgentResult(
        answer_text=answer,
        queries=queries,
        route_mode="sql",
    )
