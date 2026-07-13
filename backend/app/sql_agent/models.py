"""SQL agent runtime models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ActiveSqlSnapshot:
    """Detached copy of active connection fields safe to use after the ORM session closes."""

    connection_id: int
    display_name: str
    description: str


@dataclass(frozen=True)
class SqlToolStatus:
    """User-facing tool placeholder while the SQL agent executes internal tools."""

    name: str
    status: Literal["running", "complete"]
    label: str


@dataclass
class SqlAgentResult:
    answer_text: str = ""
    queries: list[str] = field(default_factory=list)
    connection_id: int | None = None
    display_name: str | None = None
    route_mode: str = "sql"
