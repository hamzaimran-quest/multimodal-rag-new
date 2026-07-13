"""SQL Agent connection management API."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.db.models import User, UserSqlConnection
from app.db.session import get_db
from app.sql_agent import service as sql_service

router = APIRouter(prefix="/sql-agent", tags=["sql-agent"])


class SqlConnectionSummary(BaseModel):
    id: int
    display_name: str
    description: str
    is_active: bool
    dialect: str
    last_tested_at: datetime | None = None
    last_error: str | None = None


class SqlAgentStatusResponse(BaseModel):
    has_active: bool
    active_connection: SqlConnectionSummary | None = None
    connections: list[SqlConnectionSummary] = Field(default_factory=list)


class AddSqlConnectionRequest(BaseModel):
    connection_url: str = Field(..., min_length=8)
    display_name: str = Field(..., min_length=1, max_length=120)
    description: str = Field(..., min_length=1)
    activate: bool = False


class UpdateSqlConnectionRequest(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, min_length=1)


class UpdateSqlCredentialsRequest(BaseModel):
    connection_url: str = Field(..., min_length=8)


def _summary(row: UserSqlConnection) -> SqlConnectionSummary:
    return SqlConnectionSummary(
        id=row.id,
        display_name=row.display_name,
        description=row.description,
        is_active=row.is_active,
        dialect=row.dialect,
        last_tested_at=row.last_tested_at,
        last_error=row.last_error,
    )


def _status_payload(rows: list[UserSqlConnection]) -> SqlAgentStatusResponse:
    active = next((row for row in rows if row.is_active), None)
    return SqlAgentStatusResponse(
        has_active=active is not None,
        active_connection=_summary(active) if active else None,
        connections=[_summary(row) for row in rows],
    )


@router.get("/status", response_model=SqlAgentStatusResponse)
def sql_agent_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SqlAgentStatusResponse:
    rows = sql_service.list_connections(db, current_user.id)
    return _status_payload(rows)


@router.get("/connections", response_model=list[SqlConnectionSummary])
def list_sql_connections(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[SqlConnectionSummary]:
    rows = sql_service.list_connections(db, current_user.id)
    return [_summary(row) for row in rows]


@router.post("/connections", response_model=SqlConnectionSummary, status_code=status.HTTP_201_CREATED)
def add_sql_connection(
    body: AddSqlConnectionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SqlConnectionSummary:
    try:
        row = sql_service.add_connection(
            db,
            user_id=current_user.id,
            connection_url=body.connection_url,
            display_name=body.display_name,
            description=body.description,
            activate=body.activate,
        )
        db.commit()
        db.refresh(row)
        return _summary(row)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="connection_test_failed",
        ) from exc


@router.post("/connections/{connection_id}/activate", response_model=SqlConnectionSummary)
def activate_sql_connection(
    connection_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SqlConnectionSummary:
    try:
        row = sql_service.activate_connection(db, user_id=current_user.id, connection_id=connection_id)
        db.commit()
        db.refresh(row)
        return _summary(row)
    except LookupError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="connection_not_found") from exc


@router.post("/connections/{connection_id}/test", response_model=SqlConnectionSummary)
def test_sql_connection(
    connection_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SqlConnectionSummary:
    try:
        row = sql_service.test_saved_connection(db, user_id=current_user.id, connection_id=connection_id)
        db.commit()
        db.refresh(row)
        return _summary(row)
    except LookupError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="connection_not_found") from exc
    except Exception as exc:
        db.commit()
        row = sql_service.get_connection_for_user(db, connection_id, current_user.id)
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="connection_not_found") from exc
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="connection_test_failed") from exc


@router.patch("/connections/{connection_id}", response_model=SqlConnectionSummary)
def patch_sql_connection(
    connection_id: int,
    body: UpdateSqlConnectionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SqlConnectionSummary:
    try:
        row = sql_service.update_metadata(
            db,
            user_id=current_user.id,
            connection_id=connection_id,
            display_name=body.display_name,
            description=body.description,
        )
        db.commit()
        db.refresh(row)
        return _summary(row)
    except LookupError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="connection_not_found") from exc


@router.patch("/connections/{connection_id}/credentials", response_model=SqlConnectionSummary)
def patch_sql_credentials(
    connection_id: int,
    body: UpdateSqlCredentialsRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SqlConnectionSummary:
    try:
        row = sql_service.update_credentials(
            db,
            user_id=current_user.id,
            connection_id=connection_id,
            connection_url=body.connection_url,
        )
        db.commit()
        db.refresh(row)
        return _summary(row)
    except LookupError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="connection_not_found") from exc
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="connection_test_failed") from exc


@router.delete("/connections/{connection_id}")
def delete_sql_connection(
    connection_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    deleted = sql_service.delete_connection(db, user_id=current_user.id, connection_id=connection_id)
    if not deleted:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="connection_not_found")
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/deactivate")
def deactivate_sql_connections(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    sql_service.deactivate_all(db, current_user.id)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
