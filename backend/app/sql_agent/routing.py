"""Schema-first SQL routing gate."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import UserSqlConnection
from app.llm.agent import _is_small_talk
from app.sql_agent import schema_cache
from app.sql_agent.scope_classifier import ScopeClassification, classify_query_scope

logger = logging.getLogger(__name__)

RouteMode = Literal["sql", "rag", "hybrid", "agent_only"]


@dataclass(frozen=True)
class SqlRoutePlan:
    mode: RouteMode
    schema_digest: str | None = None
    matched_tables: list[str] = field(default_factory=list)
    reason: str = ""


def _apply_confidence_gate(classification: ScopeClassification, tables: list[str]) -> RouteMode:
    decision = classification.decision
    confidence = classification.confidence

    if decision == "sql" and confidence < settings.sql_scope_min_confidence:
        if _has_entity_sql_signal(classification, tables):
            return "sql"
        return "rag"

    if decision == "sql" and not classification.matched_tables:
        entity_mode = _heuristic_entity_mode(classification.reason, tables)
        if entity_mode:
            return entity_mode

    return decision  # type: ignore[return-value]


def _has_entity_sql_signal(classification: ScopeClassification, tables: list[str]) -> bool:
    reason = classification.reason.lower()
    if "table" in reason or "schema" in reason or "film" in reason:
        return True
    table_names = {table.lower() for table in tables}
    return any(name.lower() in table_names for name in classification.matched_tables)


def _heuristic_entity_mode(reason: str, tables: list[str]) -> RouteMode | None:
    if "film" in reason.lower() and any(table.lower() == "film" for table in tables):
        return "sql"
    return None


async def plan_sql_route(
    *,
    db: Session,
    query: str,
    active_sql: UserSqlConnection | None,
    connection_url: str | None,
) -> SqlRoutePlan:
    """Decide sql/rag/hybrid using cached schema + scope classifier."""
    if active_sql is None or not connection_url:
        return SqlRoutePlan(mode="rag", reason="no_active_sql_connection")

    if _is_small_talk(query):
        return SqlRoutePlan(mode="agent_only", reason="small_talk")

    cached = schema_cache.get_or_load_schema(db, active_sql, connection_url)
    classification = await classify_query_scope(
        query=query,
        schema_digest=cached.digest,
        tables=cached.tables,
        display_name=active_sql.display_name,
        description=active_sql.description,
    )
    mode = _apply_confidence_gate(classification, cached.tables)
    logger.info(
        "SQL route_plan mode=%s reason=%r tables=%s schema_source=%s",
        mode,
        classification.reason,
        classification.matched_tables,
        cached.source,
    )
    return SqlRoutePlan(
        mode=mode,
        schema_digest=cached.digest,
        matched_tables=list(classification.matched_tables),
        reason=classification.reason,
    )
