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


def _is_sql_execute_tool(tool: Any) -> bool:
    name = str(getattr(tool, "tool", None) or tool or "")
    return name == "sql_db_query"


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().rstrip(";")).lower()


def clean_sql_answer_text(text: str) -> str:
    """Remove SQL statements or code blocks accidentally echoed in the user-facing answer."""
    cleaned = text.strip()
    cleaned = _SQL_LEADING_RE.sub("", cleaned, count=1)
    cleaned = _SQL_CODE_BLOCK_RE.sub("", cleaned)
    return cleaned.strip()


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

    db = SQLDatabase.from_uri(
        connection_url,
        sample_rows_in_table_info=0 if schema_digest else 3,
    )
    llm = build_sql_agent_llm()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    prefix = build_sql_agent_prefix(description, schema_digest=schema_digest)

    if schema_digest:
        tools = [
            tool
            for tool in toolkit.get_tools()
            if getattr(tool, "name", "") in {"sql_db_query", "sql_db_query_checker"}
        ]
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
    executor.tools = secure_sql_tools(list(executor.tools), audit=audit)
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
    answer = clean_sql_answer_text(str(response.get("output") or ""))
    steps = response.get("intermediate_steps") or []
    return SqlAgentResult(
        answer_text=answer,
        queries=_extract_queries_from_steps(steps),
        route_mode="sql",
    )
