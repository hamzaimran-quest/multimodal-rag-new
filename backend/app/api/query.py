"""Phase 5 query API: SSE streaming + final structured sources payload."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.auth.rate_limit import rate_limit_query
from app.charts.service import build_computed_charts
from app.chat import service as chat_service
from app.config import settings
from app.db.models import User
from app.db.session import get_db
from app.ingestion.embeddings import embed_texts
from app.llm.groq import stream_groq_answer
from app.llm.intent import classify_visual_intent
from app.llm.query_analyze import analyze_query
from app.opensearch.documents import get_document_for_user
from app.retrieval.docx_image_attach import (
    resolve_docx_proximity_attachments,
    retrieve_docx_intent_images,
)
from app.retrieval.image_attach import (
    build_display_images,
    resolve_proximity_attachments,
    retrieve_intent_images,
)
from app.retrieval.models import RetrievedChunk
from app.retrieval.request_log import (
    log_llm_answer,
    log_llm_context,
    log_query_stream_outcome,
    log_retrieval_request,
)
from app.retrieval.service import hybrid_retrieve

router = APIRouter(prefix="/query", tags=["query"])
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=settings.default_top_k, ge=1, le=50)
    doc_id: str | None = None
    session_id: int | None = None


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _image_source_snippet(chunk: RetrievedChunk) -> str:
    """Human-readable snippet for image sources (no indexing scaffolding)."""
    extra = chunk.extra_metadata or {}
    caption = str(extra.get("image_caption") or "").strip()
    if caption:
        return caption[:240]

    content = chunk.content or ""
    for line in content.splitlines():
        if line.startswith("Host text:"):
            return line.removeprefix("Host text:").strip()[:240]
        if line.startswith("Nearby text:"):
            return line.removeprefix("Nearby text:").strip()[:240]

    skip_prefixes = ("Part ", "Section:", "Previous text:", "Next text:", "OCR text:", "Page ")
    lines = [line for line in content.splitlines() if not any(line.startswith(p) for p in skip_prefixes)]
    text = " ".join(part.strip() for part in lines if part.strip())
    return (text or content).strip()[:240]


def _build_context(chunks: list[RetrievedChunk]) -> str:
    text_table_chunks = [c for c in chunks if c.chunk_type in {"text", "table"}]
    if not text_table_chunks:
        return "No textual/table context retrieved."

    parts: list[str] = []
    for idx, chunk in enumerate(text_table_chunks, start=1):
        parts.append(
            f"--- Source {idx} ---\n"
            f"Document: {chunk.filename}\n"
            f"Page: {chunk.page_number}\n"
            f"Type: {chunk.chunk_type}\n"
            f"Content:\n{chunk.content.strip()}"
        )
    return "\n\n".join(parts)


def _resolve_page_counts(client, chunks: list[RetrievedChunk], user_id: int) -> dict[str, int]:
    """Map each retrieved chunk's doc_id to its owner-verified page_count (single lookup per doc)."""
    counts: dict[str, int] = {}
    for doc_id in {chunk.doc_id for chunk in chunks}:
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
        snippet = (
            _image_source_snippet(chunk)
            if chunk.chunk_type == "image"
            else chunk.content[:240]
        )
        extra = chunk.extra_metadata or {}
        viewer = extra.get("viewer_location") or {}
        source_format = extra.get("source_format", "pdf")

        if source_format == "docx":
            bbox = viewer.get("bbox") if viewer.get("match_status") == "ok" else None
            line_bboxes = viewer.get("line_bboxes") if viewer.get("match_status") == "ok" else None
            viewer_page = viewer.get("viewer_page")
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
            }
        )
    return sources


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
                "source_format": image.get("source_format", "pdf"),
                "section": image.get("section"),
                "bbox": image.get("bbox"),
                "line_bboxes": None,
                "page_count": page_counts.get(image.get("doc_id"), 0),
                "attached_images": [],
                "attach_reason": "intent",
            }
        )


def _enrich_docx_image_sources_with_viewer_page(sources: list[dict]) -> None:
    """Copy viewer_page from nearby DOCX text/table citations onto image sources."""
    viewer_by_block: dict[tuple[str, int], int] = {}
    for source in sources:
        if source.get("source_format") != "docx":
            continue
        if source.get("chunk_type") not in {"text", "table"}:
            continue
        viewer_page = source.get("viewer_page")
        if viewer_page:
            viewer_by_block[(source["doc_id"], source["page_number"])] = viewer_page

    window = settings.docx_proximity_block_window
    for source in sources:
        if source.get("chunk_type") != "image" or source.get("source_format") != "docx":
            continue
        if source.get("viewer_page"):
            continue
        doc_id = source.get("doc_id")
        block = source.get("page_number")
        if not doc_id or not block:
            continue
        viewer_page = None
        for offset in range(window + 1):
            candidates = [block] if offset == 0 else [block - offset, block + offset]
            for candidate in candidates:
                if candidate <= 0:
                    continue
                viewer_page = viewer_by_block.get((doc_id, candidate))
                if viewer_page:
                    break
            if viewer_page:
                break
        if viewer_page:
            source["viewer_page"] = viewer_page


@router.post("/stream")
async def stream_query(
    request: Request,
    body: QueryRequest,
    current_user: User = Depends(rate_limit_query),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    client = request.app.state.opensearch

    if body.doc_id is not None:
        owned = get_document_for_user(client, body.doc_id, current_user.id)
        if owned is None:
            raise HTTPException(status_code=404, detail="Document not found")

    try:
        chat = chat_service.resolve_session_for_query(
            db,
            user_id=current_user.id,
            session_id=body.session_id,
            first_message=body.query,
        )
    except LookupError:
        raise HTTPException(status_code=404, detail="Chat not found") from None

    chat_service.append_user_message(db, chat, body.query)
    db.commit()
    session_id = chat.id

    history = chat_service.recent_history(db, chat, limit=settings.chat_history_turns)

    # Follow-up turns: query writer rewrites ambiguous messages (pronouns, "these",
    # omitted subjects) into a standalone search query using recent history, and
    # classifies visual intent in the same call. First turns skip this and run
    # intent in parallel with retrieval to stay fast.
    if history:
        analysis = await analyze_query(history, body.query)
        search_query = analysis.get("standalone_query") or body.query
        intent = {
            "visual_intent": analysis.get("visual_intent", "none"),
            "confidence": analysis.get("confidence", 0.0),
        }
        if not settings.image_intent_enabled:
            intent = {"visual_intent": "none", "confidence": 0.0}
        intent_task = None
    else:
        search_query = body.query
        intent = {"visual_intent": "none", "confidence": 0.0}
        intent_task = asyncio.create_task(classify_visual_intent(body.query))

    try:
        retrieval = await asyncio.to_thread(
            hybrid_retrieve,
            client,
            search_query,
            user_id=current_user.id,
            top_k=body.top_k,
            doc_id=body.doc_id,
        )
    except Exception as exc:
        if intent_task is not None:
            intent_task.cancel()
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    if intent_task is not None:
        try:
            intent = await intent_task
        except Exception:
            intent = {"visual_intent": "none", "confidence": 0.0}

    page_counts = _resolve_page_counts(client, retrieval.results, current_user.id)
    context = _build_context(retrieval.results)

    # Image attachment (PDF proximity + DOCX block proximity + intent; fail-closed inside).
    attachments: dict[str, list[dict]] = {}
    intent_images: list[dict] = []
    if settings.image_attach_enabled:
        attachments = resolve_proximity_attachments(
            client, retrieval.results, user_id=current_user.id
        )
        docx_attachments = resolve_docx_proximity_attachments(
            client, retrieval.results, user_id=current_user.id
        )
        for anchor_id, images in docx_attachments.items():
            attachments.setdefault(anchor_id, []).extend(images)
        if intent.get("visual_intent") == "required":
            try:
                query_vector = embed_texts([search_query])[0]
                pdf_intent_images = retrieve_intent_images(
                    client,
                    search_query,
                    query_vector,
                    user_id=current_user.id,
                    doc_id=body.doc_id,
                    text_chunks=retrieval.results,
                )
                docx_intent_images = retrieve_docx_intent_images(
                    client,
                    search_query,
                    query_vector,
                    user_id=current_user.id,
                    doc_id=body.doc_id,
                )
                intent_images = sorted(
                    pdf_intent_images + docx_intent_images,
                    key=lambda image: image.get("score", 0.0),
                    reverse=True,
                )
                # Explicit requests get the single best match, not a montage.
                intent_images = intent_images[: settings.image_intent_max_display]
            except Exception:
                logger.warning("Intent image retrieval failed", exc_info=True)

    sources = _build_sources(retrieval.results, page_counts, attachments)
    _merge_intent_image_sources(sources, intent_images, page_counts)
    _enrich_docx_image_sources_with_viewer_page(sources)
    display_images = build_display_images(intent_images, attachments)

    # Keep image handling in sources/UI only; avoid steering LLM to answer with
    # "Here's an image ..." phrasing.
    visual_note: str | None = None
    logger.info(
        "IMAGE_ATTACH query_preview=%r visual_intent=%s intent_images=%s "
        "proximity_anchors=%s hero_images=%s",
        body.query[:80],
        intent.get("visual_intent"),
        len(intent_images),
        len(attachments),
        len(display_images),
    )
    charts = build_computed_charts(retrieval.results)
    log_retrieval_request(
        endpoint="/query/stream",
        query=search_query,
        top_k=body.top_k,
        doc_id=body.doc_id,
        chunks=retrieval.results,
        charts=charts,
    )
    log_llm_context(query=body.query, context=context, source_count=len(sources))

    async def event_stream() -> AsyncGenerator[str, None]:
        yield _sse(
            "meta",
            {
                "query": body.query,
                "top_k": body.top_k,
                "doc_id": body.doc_id,
                "session_id": session_id,
            },
        )
        answer_parts: list[str] = []
        try:
            async for token in stream_groq_answer(
                query=body.query, context=context, history=history, visual_note=visual_note
            ):
                answer_parts.append(token)
                yield _sse("token", {"token": token})
            yield _sse("sources", {"sources": sources})
            if charts:
                yield _sse("charts", {"charts": charts})
            yield _sse("done", {"ok": True})
            log_llm_answer(query=body.query, answer="".join(answer_parts))
            log_query_stream_outcome(query=body.query, ok=True)

            chat_service.append_assistant_message(
                db,
                chat,
                "".join(answer_parts),
                sources=sources,
                charts=charts,
            )
            db.commit()
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})
            yield _sse("done", {"ok": False})
            log_query_stream_outcome(query=body.query, ok=False, error=str(exc))

    return StreamingResponse(event_stream(), media_type="text/event-stream")
