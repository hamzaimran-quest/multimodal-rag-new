"""Phase 5 query API: SSE streaming + final structured sources payload."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, Iterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.auth.rate_limit import rate_limit_query
from app.charts.build import merge_chart_outputs
from app.charts.auto import try_auto_chart_from_retrieval
from app.chat import service as chat_service
from app.config import settings
from app.db.models import User
from app.db.session import SessionLocal, get_db
from app.retrieval.context import build_llm_context, select_chunks_for_llm_context
from app.llm.agent import AgentTurnResult, iter_agent_turn
from app.llm.groq import stream_groq_answer
from app.opensearch.documents import get_document_for_user
from app.retrieval.image_attach import build_display_images, resolve_proximity_attachments
from app.retrieval.models import RetrievedChunk
from app.retrieval.request_log import (
    log_llm_answer,
    log_llm_context,
    log_query_stream_outcome,
    log_retrieval_request,
)
from app.ingestion.xlsx_highlight import apply_xlsx_highlights_to_sources
from app.retrieval.scope import scope_filenames, validate_scope_doc_ids

router = APIRouter(prefix="/query", tags=["query"])
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=settings.default_top_k, ge=1, le=50)
    doc_id: str | None = None
    doc_ids: list[str] | None = None
    session_id: int | None = None


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _resolve_page_counts(
    client,
    chunks: list[RetrievedChunk],
    user_id: int,
    *,
    extra_doc_ids: set[str] | None = None,
) -> dict[str, int]:
    """Map each doc_id to its owner-verified page_count (single lookup per doc)."""
    counts: dict[str, int] = {}
    doc_ids = {chunk.doc_id for chunk in chunks if chunk.doc_id}
    if extra_doc_ids:
        doc_ids.update(extra_doc_ids)
    for doc_id in doc_ids:
        record = get_document_for_user(client, doc_id, user_id)
        if record is not None:
            counts[doc_id] = int(record.get("page_count", 0) or 0)
    return counts


def _build_sources(
    chunks: list[RetrievedChunk],
    page_counts: dict[str, int] | None = None,
    attachments: dict[str, list[dict]] | None = None,
) -> list[dict]:
    page_counts = page_counts or {}
    attachments = attachments or {}
    sources: list[dict] = []
    for chunk in chunks:
        snippet = chunk.content[:240]
        extra = chunk.extra_metadata or {}
        viewer = extra.get("viewer_location") or {}
        source_format = extra.get("source_format", "pdf")

        if source_format == "docx":
            bbox = viewer.get("bbox") if viewer.get("match_status") == "ok" else None
            line_bboxes = viewer.get("line_bboxes") if viewer.get("match_status") == "ok" else None
            viewer_page = viewer.get("viewer_page")
        elif source_format == "xlsx":
            bbox = None
            line_bboxes = None
            viewer_page = None
            row_range = extra.get("row_range")
        else:
            bbox = chunk.bbox
            line_bboxes = extra.get("line_bboxes")
            viewer_page = None

        sources.append(
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "viewer_page": viewer_page,
                "chunk_type": chunk.chunk_type,
                "snippet": snippet,
                "image_url": chunk.image_url,
                "score": chunk.score,
                "source_format": source_format,
                "section": extra.get("section"),
                "bbox": bbox,
                "line_bboxes": line_bboxes,
                "page_count": page_counts.get(chunk.doc_id, 0),
                "attached_images": attachments.get(chunk.chunk_id, []),
                "sheet_name": extra.get("sheet_name"),
                "sheet_index": extra.get("sheet_index"),
                "sheet_role": extra.get("sheet_role") if source_format == "xlsx" else None,
                "row_range": row_range if source_format == "xlsx" else extra.get("row_range"),
                "col_range": extra.get("col_range"),
            }
        )
    return sources


def _build_chart_note(charts: list[dict]) -> str | None:
    """Tell the answer model a chart is already shown in the UI."""
    if not charts:
        return None
    chart = charts[0]
    chart_type = str(chart.get("chart_type") or "chart").strip()
    title = str(chart.get("title") or "").strip()
    title_hint = f" Title: {title}." if title else ""
    return (
        f"A {chart_type} chart is already rendered separately in the Charts panel "
        f"for this reply.{title_hint} "
        "Do not include matplotlib, Python, JavaScript, or other plotting code. "
        "Do not draw ASCII art or recreate the chart in text. "
        "Reply in one or two short sentences summarizing the trend or key values only."
    )


def _build_visual_note(display_images: list[dict]) -> str | None:
    """Tell the answer model an image is shown in the UI (not in text excerpts)."""
    if not display_images:
        return None
    caption = str(display_images[0].get("caption") or "").strip()
    caption_hint = f" Nearby caption: {caption}." if caption else ""
    return (
        "The user asked to see an image from the documents; a matching image is "
        f"displayed separately in the UI (not in the excerpts above).{caption_hint} "
        "Do not say it was not found. Reply in one short sentence; name the subject "
        "from the excerpts when possible."
    )


def _merge_intent_image_sources(
    sources: list[dict],
    intent_images: list[dict],
    page_counts: dict[str, int],
) -> None:
    """Add intent-retrieved images as image sources (persisted) so explicit visual
    requests survive chat-history reload. Existing image sources are tagged instead
    of duplicated."""
    by_id = {s["chunk_id"]: s for s in sources}
    for image in intent_images:
        image_id = image["image_chunk_id"]
        existing = by_id.get(image_id)
        if existing is not None:
            existing["attach_reason"] = "intent"
            continue
        sources.append(
            {
                "chunk_id": image_id,
                "doc_id": image.get("doc_id"),
                "filename": image.get("filename"),
                "page_number": image.get("page_number"),
                "viewer_page": None,
                "chunk_type": "image",
                "snippet": image.get("caption") or "",
                "image_url": image.get("image_url"),
                "score": image.get("score", 0.0),
                "source_format": "pdf",
                "section": None,
                "bbox": image.get("bbox"),
                "line_bboxes": None,
                "page_count": page_counts.get(image.get("doc_id"), 0),
                "attached_images": [],
                "attach_reason": "intent",
            }
        )


def _tokenize_for_sse(text: str) -> Iterator[str]:
    """Emit direct agent replies through the same token SSE shape."""
    if not text:
        return
    parts = text.split(" ")
    for idx, part in enumerate(parts):
        yield part if idx == 0 else f" {part}"


def _assemble_retrieval_payload(
    client,
    chunks: list[RetrievedChunk],
    *,
    user_id: int,
    intent_images: list[dict] | None = None,
    visual_intent_required: bool = False,
    tool_charts: list[dict] | None = None,
) -> tuple[str, list[dict], list[dict], str | None, list[RetrievedChunk]]:
    """Shared post-retrieval enrichment: context, sources, charts, optional visual note."""
    intent_images = intent_images or []
    attachments: dict[str, list[dict]] = {}
    if settings.image_attach_enabled:
        attachments = resolve_proximity_attachments(client, chunks, user_id=user_id)

    if visual_intent_required and intent_images:
        intent_images = sorted(intent_images, key=lambda i: i["score"], reverse=True)[:1]

    hero_cap = 1 if visual_intent_required else settings.image_max_display
    display_images = build_display_images(intent_images, attachments, max_display=hero_cap)
    visual_note = _build_visual_note(display_images) if visual_intent_required else None

    page_counts = _resolve_page_counts(
        client,
        chunks,
        user_id,
        extra_doc_ids={img.get("doc_id") for img in intent_images if img.get("doc_id")},
    )
    grounding_chunks = select_chunks_for_llm_context(chunks)
    sources = _build_sources(grounding_chunks, page_counts, attachments)
    _merge_intent_image_sources(sources, intent_images, page_counts)
    charts = merge_chart_outputs(tool_charts or [])
    context = build_llm_context(grounding_chunks)
    return context, sources, charts, visual_note, grounding_chunks


def _log_agent_chunks(query: str, tools: list[str], chunks: list[RetrievedChunk]) -> None:
    by_type: dict[str, int] = {}
    for chunk in chunks:
        by_type[chunk.chunk_type] = by_type.get(chunk.chunk_type, 0) + 1
    logger.info(
        "AGENT chunks_retrieved query_preview=%r tools=%s total=%s by_type=%s sources_emitted=%s",
        query[:80],
        tools,
        len(chunks),
        by_type,
        len(chunks),
    )
    for index, chunk in enumerate(chunks, start=1):
        logger.info(
            "AGENT chunk[%s] id=%s type=%s file=%s page=%s score=%.4f chars=%s preview=%r",
            index,
            chunk.chunk_id,
            chunk.chunk_type,
            chunk.filename,
            chunk.page_number,
            chunk.score,
            len(chunk.content),
            chunk.content[:100].replace("\n", " "),
        )


def _persist_assistant_reply(
    chat_id: int,
    content: str,
    *,
    sources: list[dict] | None = None,
    charts: list[dict] | None = None,
) -> None:
    """Save assistant message using a fresh session (SSE runs after request scope ends)."""
    from app.db.models import ChatSession

    db = SessionLocal()
    try:
        chat = db.get(ChatSession, chat_id)
        if chat is None:
            logger.error("persist_assistant_reply chat_not_found chat_id=%s", chat_id)
            return
        chat_service.append_assistant_message(
            db,
            chat,
            content,
            sources=sources,
            charts=charts,
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("persist_assistant_reply failed chat_id=%s", chat_id)
        raise
    finally:
        db.close()


@router.post("/stream")
async def stream_query(
    request: Request,
    body: QueryRequest,
    current_user: User = Depends(rate_limit_query),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    client = request.app.state.opensearch
    user_id = current_user.id

    scope_doc_ids = validate_scope_doc_ids(
        client,
        user_id=user_id,
        doc_ids=body.doc_ids,
        doc_id=body.doc_id,
    )
    scoped_filenames = (
        scope_filenames(client, user_id=user_id, scope_doc_ids=scope_doc_ids)
        if scope_doc_ids
        else None
    )

    try:
        chat = chat_service.resolve_session_for_query(
            db,
            user_id=user_id,
            session_id=body.session_id,
            first_message=body.query,
        )
    except LookupError:
        raise HTTPException(status_code=404, detail="Chat not found") from None

    chat_service.append_user_message(db, chat, body.query)
    db.commit()
    session_id = chat.id
    chat_id = chat.id

    last_assistant_reply = chat_service.latest_assistant_reply(
        db,
        chat,
        max_chars=settings.chat_last_reply_max_chars,
        exclude_last_user=True,
    )

    prior_queries = chat_service.prior_user_queries(
        db,
        chat,
        max_turns=settings.chat_history_turns,
        exclude_last_user=True,
    )
    prior_table_chunk_ids = chat_service.prior_table_chunk_ids(
        db,
        chat,
        exclude_last_user=True,
    )
    return StreamingResponse(
        _agent_event_stream(
            body=body,
            chat_id=chat_id,
            session_id=session_id,
            user_id=user_id,
            client=client,
            prior_queries=prior_queries,
            last_assistant_reply=last_assistant_reply,
            prior_table_chunk_ids=prior_table_chunk_ids,
            scope_doc_ids=scope_doc_ids,
            scoped_filenames=scoped_filenames,
        ),
        media_type="text/event-stream",
    )


async def _agent_event_stream(
    *,
    body: QueryRequest,
    chat_id: int,
    session_id: int,
    user_id: int,
    client,
    prior_queries: list[str],
    last_assistant_reply: str | None,
    prior_table_chunk_ids: list[str],
    scope_doc_ids: list[str] | None,
    scoped_filenames: list[str] | None,
) -> AsyncGenerator[str, None]:
    yield _sse(
        "meta",
        {
            "query": body.query,
            "top_k": body.top_k,
            "doc_id": scope_doc_ids[0] if scope_doc_ids and len(scope_doc_ids) == 1 else None,
            "doc_ids": scope_doc_ids,
            "session_id": session_id,
            "agent": True,
        },
    )

    turn: AgentTurnResult | None = None
    try:
        async for event in iter_agent_turn(
            client,
            user_id=user_id,
            user_query=body.query,
            prior_queries=prior_queries,
            last_assistant_reply=last_assistant_reply,
            prior_table_chunk_ids=prior_table_chunk_ids,
            scope_doc_ids=scope_doc_ids,
            scoped_filenames=scoped_filenames,
            default_top_k=body.top_k,
        ):
            if event["type"] == "tool":
                yield _sse(
                    "tool",
                    {
                        "name": event["name"],
                        "status": event["status"],
                        "round": event.get("round"),
                    },
                )
            elif event["type"] == "complete":
                turn = event["result"]
    except Exception as exc:
        logger.exception(
            "AGENT turn_failed user_id=%s session_id=%s query_preview=%r",
            user_id,
            session_id,
            body.query[:120],
        )
        yield _sse("error", {"message": f"Agent failed: {exc}"})
        yield _sse("done", {"ok": False})
        return

    assert turn is not None
    logger.info(
        "AGENT flow_selected session_id=%s direct=%s clarification=%s tools=%s rounds=%s",
        session_id,
        turn.direct_answer is not None,
        turn.is_clarification,
        turn.tools_used,
        turn.rounds_used,
    )

    visual_intent_required = "search_images" in turn.tools_used
    tool_charts = list(turn.tool_charts)
    chart_note: str | None = None
    if "create_chart" not in turn.tools_used:
        auto_charts, chart_note = try_auto_chart_from_retrieval(
            client,
            user_id=user_id,
            user_query=body.query,
            retrieved_chunks=turn.retrieved_chunks,
            scope_doc_ids=scope_doc_ids,
            prior_table_chunk_ids=prior_table_chunk_ids,
        )
        tool_charts = merge_chart_outputs(tool_charts, auto_charts)

    context, sources, charts, visual_note, grounding_chunks = _assemble_retrieval_payload(
        client,
        turn.retrieved_chunks,
        user_id=user_id,
        intent_images=turn.intent_images,
        visual_intent_required=visual_intent_required,
        tool_charts=tool_charts,
    )
    if charts:
        chart_note = _build_chart_note(charts)
    _log_agent_chunks(body.query, turn.tools_used, turn.retrieved_chunks)

    if turn.retrieved_chunks:
        log_retrieval_request(
            endpoint="/query/stream",
            query=body.query,
            top_k=body.top_k,
            doc_id=scope_doc_ids[0] if scope_doc_ids and len(scope_doc_ids) == 1 else None,
            chunks=turn.retrieved_chunks,
            charts=charts,
        )
        log_llm_context(query=body.query, context=context, source_count=len(sources))
    elif turn.tools_used:
        log_llm_context(query=body.query, context=context, source_count=0)

    logger.info(
        "AGENT query_preview=%r tools=%s chunks=%s intent_images=%s",
        body.query[:80],
        turn.tools_used,
        len(turn.retrieved_chunks),
        len(turn.intent_images),
    )

    answer_parts: list[str] = []
    try:
        if turn.direct_answer is not None:
            logger.info("AGENT answer_stream mode=direct chars=%s", len(turn.direct_answer))
            for token in _tokenize_for_sse(turn.direct_answer):
                answer_parts.append(token)
                yield _sse("token", {"token": token})
        else:
            logger.info(
                "AGENT answer_stream mode=grounded context_chars=%s sources=%s",
                len(context),
                len(sources),
            )
            async for token in stream_groq_answer(
                query=body.query,
                context=context,
                visual_note=visual_note,
                chart_note=chart_note,
                last_assistant_reply=last_assistant_reply,
            ):
                answer_parts.append(token)
                yield _sse("token", {"token": token})

        answer = "".join(answer_parts)
        highlight_updated = apply_xlsx_highlights_to_sources(
            sources,
            grounding_chunks,
            query=body.query,
            answer=answer,
        )
        logger.info(
            "XLSX_HIGHLIGHT query_apply updated=%s source_count=%s grounding_chunks=%s",
            highlight_updated,
            len(sources),
            len(grounding_chunks),
        )
        yield _sse("sources", {"sources": sources})
        if charts:
            yield _sse("charts", {"charts": charts})
        yield _sse("done", {"ok": True})
        log_llm_answer(query=body.query, answer=answer)
        log_query_stream_outcome(query=body.query, ok=True)

        _persist_assistant_reply(
            chat_id,
            answer,
            sources=sources,
            charts=charts,
        )
    except Exception as exc:
        yield _sse("error", {"message": str(exc)})
        yield _sse("done", {"ok": False})
        log_query_stream_outcome(query=body.query, ok=False, error=str(exc))
