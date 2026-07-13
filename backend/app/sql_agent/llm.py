"""LLM client for the LangChain SQL agent (OpenRouter or Groq fallback)."""

from __future__ import annotations

from app.config import settings


def build_sql_agent_llm():
    """Return a streaming chat model for the SQL agent executor."""
    if settings.sql_agent_openrouter_configured:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.resolved_sql_agent_model,
            api_key=settings.sql_agent_openrouter_api_key,
            base_url=settings.sql_agent_openrouter_base_url,
            temperature=0,
            streaming=True,
        )

    if not settings.groq_configured:
        raise RuntimeError(
            "SQL agent LLM is not configured. Set SQL_AGENT_OPENROUTER_API_KEY "
            "or GROQ_API_KEY."
        )

    from langchain_groq import ChatGroq

    return ChatGroq(
        model=settings.resolved_sql_agent_model,
        groq_api_key=settings.groq_api_key,
        temperature=0,
        streaming=True,
    )
