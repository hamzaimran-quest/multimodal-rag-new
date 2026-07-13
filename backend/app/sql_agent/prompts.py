"""Prompt helpers for the LangChain SQL agent."""

from __future__ import annotations


def build_sql_agent_prefix(description: str, *, schema_digest: str | None = None) -> str:
    desc = description.strip()
    prefix = (
        "You are a PostgreSQL data assistant. Answer questions by running read-only SELECT queries only. "
        "Never modify data.\n\n"
        f"Database context (provided by the user): {desc}\n"
        "Use this context to interpret table and column names and business meaning.\n\n"
        "Answer formatting rules:\n"
        "- Never include SQL statements, query text, or code blocks in your final answer.\n"
        "- Report only facts returned by executed queries. Do not invent placeholder rows or example names.\n"
        "- If related data was not queried, do not guess or offer sample values; say it was not retrieved.\n"
        "- Be concise and direct. Do not add follow-up suggestions unless the user asked for them."
    )
    if schema_digest:
        prefix += (
            "\n\nCached schema (authoritative — do not call schema listing tools):\n"
            f"{schema_digest}\n"
            "Use sql_db_query directly. Do not call sql_db_list_tables or sql_db_schema."
        )
    return prefix
