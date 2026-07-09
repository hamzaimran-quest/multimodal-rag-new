# Multimodal RAG — Workflow Guide

This guide explains how the retrieval-augmented generation (RAG) pipeline works in this project: how **PDF and DOCX** documents become searchable chunks, how embeddings and OpenSearch power retrieval, and how text, tables, and images are handled differently at ingest and query time.

For original design decisions and open questions, see [ARCHITECTURE.md](./ARCHITECTURE.md).

---

## Table of contents

1. [High-level overview](#1-high-level-overview)
2. [End-to-end data flow](#2-end-to-end-data-flow)
3. [Embeddings](#3-embeddings)
4. [OpenSearch](#4-opensearch)
5. [Ingestion pipeline](#5-ingestion-pipeline)
6. [Modality handling: text, tables, images](#6-modality-handling-text-tables-images)
7. [Retrieval (hybrid search)](#7-retrieval-hybrid-search)
8. [Query and answer generation](#8-query-and-answer-generation)
9. [Computed charts (from table chunks)](#9-computed-charts-from-table-chunks)
10. [Authentication, signup, and sessions](#10-authentication-signup-and-sessions)
11. [Per-user document isolation](#11-per-user-document-isolation)
12. [Chat history](#12-chat-history)
13. [Security hardening](#13-security-hardening)
14. [API reference](#14-api-reference)
15. [Configuration](#15-configuration)
16. [Project layout](#16-project-layout)
17. [Running locally](#17-running-locally)
18. [Tuning and debugging](#18-tuning-and-debugging)
19. [PDF citation viewer and highlighting](#19-pdf-citation-viewer-and-highlighting)

---

## 1. High-level overview

This is a **multimodal document Q&A system**. It:

1. **Ingests** PDFs (page-by-page) and DOCX files (body-order), extracting text, tables, and (PDF only) images/charts.
2. **Chunks** content into retrievable units (one table = one chunk; text split with overlap).
3. **Embeds** each chunk's `content` field with a local sentence-transformer model.
4. **Indexes** chunks in **OpenSearch** with both BM25 (keyword) and k-NN (vector) fields.
5. **Retrieves** the top-k relevant chunks using **hybrid search** (BM25 + vector, score-normalized) — either always-on (legacy) or via an **optional Groq tool-calling agent** that decides when to search.
6. **Generates** grounded answers via **Groq** (`llama-3.3-70b-versatile`), streamed over SSE — answers use **retrieved excerpts only**, never router world knowledge.
7. **Cites sources** (filename, location, chunk type, snippet) and opens a **PDF citation viewer** with per-line highlight overlays.
8. **Optionally renders computed charts** when a chartable **table** chunk is retrieved (bar/line from parsed table numbers — separate from PDF images).
9. **Authenticates users** via email/password signup; each account has isolated documents, chats, and refresh sessions stored in **PostgreSQL**.

```
┌─────────────┐  signup/login   ┌──────────────┐
│  Frontend   │ ───────────────► │ PostgreSQL   │  users, refresh tokens, chat sessions
│  (React)    │                 └──────────────┘
└──────┬──────┘
       │  upload (Bearer)         ┌──────────────────┐     index      ┌─────────────┐
       ├────────────────────────► │ Ingestion        │ ─────────────► │ OpenSearch  │
       │                          │ Pipeline         │  user-scoped   │ rag_chunks  │
       │                          └──────────────────┘                └──────┬──────┘
       │  query (SSE, Bearer)                                                    │
       ▼                                                                         │
┌─────────────┐     embed + hybrid search (user_id filter)              ◄────────┘
│  Query API  │ ────────────────────────────────────────────────────────────┘
└──────┬──────┘
       │  context (text/table only)              chart spec (table chunks)
       ▼                                              │
┌─────────────┐                                       ▼
│  Groq LLM   │ ──► answer + sources + optional computed charts (UI)
└─────────────┘
       │
       └──► chat messages persisted to PostgreSQL (per session)
```

**Key design choices:**

- **PDF image chunks** are retrieved via OCR + proximity text but **not sent to the LLM**; pixels render in the Sources panel (loaded via authenticated `/images` API).
- **PDF citation highlighting** uses stored **per-line bounding boxes** (`line_bboxes`) in PDF coordinate space; the side-panel viewer renders marker-style overlays scaled to zoom/panel width (see [§19](#19-pdf-citation-viewer-and-highlighting)).
- **DOCX ingestion** (Phase 1: text + tables) uses native `python-docx` parsing; DOCX sources show **section names** in citations, not page numbers. PDF viewer is PDF-only.
- **Computed charts** are derived from **table markdown** at query time; no LLM generates chart values. A chart appears only when hybrid search retrieves a table chunk marked `chartable` at ingestion — **never** from query keywords like "chart" or "graph".
- **Per-user isolation** — every document, chunk, search, and chat is scoped by `user_id` from the JWT; cross-tenant access returns **404** (not 403).
- **Access token in memory** on the frontend; refresh token in an **httpOnly cookie** scoped to `/auth` only.
- **Agent mode** (`AGENT_ENABLED=true`) — Groq router calls tools (`search_documents`, `search_images`, `list_documents`) across up to `AGENT_MAX_ROUNDS` turns before a grounded answer stream; legacy mode always runs hybrid search on every query.

---

## 2. End-to-end data flow

### Upload → index

| Step | What happens | Code |
|------|--------------|------|
| 1 | Authenticated user uploads **PDF or DOCX** via `POST /documents/upload` (`Authorization: Bearer`) | `backend/app/api/documents.py` |
| 2 | Backend assigns `doc_id`, saves file to `data/uploads/{user_id}/{doc_id}/` | `save_upload_file()` |
| 3 | Document record created in `rag_documents` with `user_id` (`status: processing`) | `create_document_record()` |
| 4 | Ingestion runs in a FastAPI `BackgroundTasks` worker | `_schedule_ingestion()` |
| 5 | Format dispatch: PDF → `extract_page_chunks()` per page; DOCX → `extract_docx_chunks()` body-order | `pipeline.py`, `text.py`, `docx_extract.py` |
| 6 | All chunk `content` strings are batch-embedded | `embed_texts()` |
| 7 | Old chunks for this `doc_id` deleted; new chunks indexed with `user_id` | `index_chunks()` |
| 8 | Document status updated to `indexed` with `chunk_count` and (PDF) `page_count` | `update_document_record()` |

### Query → answer

Two paths controlled by **`AGENT_ENABLED`** (see [§8](#8-query-and-answer-generation)).

#### Shared steps (both paths)

| Step | What happens | Code |
|------|--------------|------|
| 1 | Authenticated user sends question via `POST /query/stream` (optional `session_id`, optional `doc_id`) | `backend/app/api/query.py` |
| 2 | User message appended to chat session in PostgreSQL (session created if omitted) | `chat/service.py` |
| 3 | SSE stream begins; assistant reply persisted after stream completes (fresh DB session) | `_persist_assistant_reply()` |

#### Legacy path (`AGENT_ENABLED=false`)

| Step | What happens | Code |
|------|--------------|------|
| 4 | Query embedded; **always** runs hybrid search | `hybrid_retrieve()` |
| 5 | Parallel visual-intent classifier for explicit image requests | `classify_visual_intent()` → `retrieve_intent_images()` |
| 6 | Text/table context assembled for LLM; proximity image attach | `_assemble_retrieval_payload()` |
| 7 | Groq streams grounded tokens (`event: token`) | `stream_groq_answer()` |
| 8 | Sources + optional charts SSE events | `_build_sources()`, `build_computed_charts()` |

#### Agent path (`AGENT_ENABLED=true`)

| Step | What happens | Code |
|------|--------------|------|
| 4 | Recent chat history sent to **router** Groq (tool-calling, up to `AGENT_MAX_ROUNDS`) | `iter_agent_turn()` in `llm/agent.py` |
| 5 | Router may call `search_documents`, `search_images`, `list_documents` (multi-round refine) | `llm/tools.py` |
| 6 | Tool progress emitted live (`event: tool` running/complete) | `_agent_event_stream()` |
| 7 | Greetings/clarification → direct router reply; document Q&A → **grounded** answer from chunks only | `stream_groq_answer()` (no chat history in answer step) |
| 8 | Forced `search_documents` fallback if router skips tools on a factual question | `_force_search_documents()` + `rewrite_retrieval_query()` for pronoun follow-ups |
| 9 | Sources + optional charts from merged retrieval results | `_assemble_retrieval_payload()` |

---

## 3. Embeddings

**Location:** `backend/app/ingestion/embeddings.py`

### Model

| Setting | Default | Notes |
|---------|---------|-------|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Resolved to `sentence-transformers/all-MiniLM-L6-v2` in FastEmbed |
| `EMBEDDING_DIMENSION` | `384` | Must match OpenSearch `knn_vector` mapping |
| Runtime | FastEmbed (ONNX) | No PyTorch; model loaded once via `@lru_cache` |

### Process

1. `get_embedding_model()` loads the FastEmbed `TextEmbedding` model on first use.
2. `embed_texts(texts)` calls `model.embed(texts)` for a batch of strings.
3. Each vector is **explicitly L2-normalized** to unit length (matching sentence-transformers `normalize_embeddings=True` behavior).
4. `assert_unit_vectors()` validates norms are ≈ 1.0 before returning.

```python
# Simplified flow
raw_vectors = model.embed(texts)
vectors = [l2_normalize(v) for v in raw_vectors]
```

### Where embeddings are created

| When | Input embedded |
|------|----------------|
| **Ingestion** | Every chunk's `content` field (text markdown, table markdown, or image OCR/proximity text) |
| **Retrieval** | The user's natural-language query |

**Critical:** Ingest and query must use the **same model and normalization**. Changing `EMBEDDING_MODEL` or `EMBEDDING_DIMENSION` requires recreating the OpenSearch index.

---

## 4. OpenSearch

**Location:** `backend/app/opensearch/`

### Infrastructure

- **Docker image:** `opensearchproject/opensearch:2.19.0` (single-node, security disabled for local dev)
- **k-NN:** HNSW index with L2 space (`engine: lucene`)
- **Bootstrap on startup:** `wait_for_opensearch()` → `ensure_indices()` → `ensure_hybrid_search_pipeline()`
- **Legacy wipe:** indices missing `user_id` in mapping are deleted on startup; `data/uploads` and `data/images` are cleared

### Two indices

#### `rag_chunks` — searchable chunk store

One OpenSearch document per chunk. Schema (from `chunks_index_body()`):

| Field | Type | Purpose |
|-------|------|---------|
| `chunk_id` | keyword | UUID per chunk |
| `doc_id` | keyword | Parent document; delete-cascade key |
| `user_id` | keyword | Owner user ID (denormalized for search filters) |
| `filename` | keyword | Original document name |
| `page_number` | integer | PDF: 1-based page. DOCX: 1-based block ordinal in reading order |
| `chunk_type` | keyword | `text`, `table`, or `image` |
| `content` | text (standard analyzer) | Human-readable chunk; BM25 target + embedding source |
| `embedding` | knn_vector (384-dim, HNSW/L2) | Vector similarity target |
| `bbox` | float[] | Union bounding box in **PDF coordinate space** (`[x0, top, x1, bottom]`, top-origin). Used for scroll/centering and fallback highlight |
| `image_path` | keyword | Filesystem path for image chunks |
| `upload_timestamp` | date | When indexed |
| `extra_metadata` | object | `extraction_method`, `chart_profile`, `line_bboxes`, `source_format`, `section`, table QA, OCR flags |

#### `rag_documents` — ingestion status registry

Tracks upload/ingestion progress per document (not used for search):

| Field | Purpose |
|-------|---------|
| `doc_id`, `filename` | Identity |
| `user_id` | Owner user ID |
| `ingestion_status` | `pending` \| `processing` \| `indexed` \| `failed` |
| `ingestion_progress` | 0–100 float for UI progress ring |
| `progress_message` | Human-readable stage |
| `chunk_count` | Set when indexing completes |
| `page_count` | Total PDF pages (set at ingestion; `0` for DOCX) |
| `error_message` | Set on failure |

### Hybrid search pipeline

**ID:** `hybrid-search-pipeline` (configurable via `HYBRID_SEARCH_PIPELINE`)

Defined in `backend/app/opensearch/pipelines.py`:

- **Normalization:** min-max per score type
- **Combination:** arithmetic mean with weights `[0.5, 0.5]` (BM25 and k-NN equal)

Applied via the `search_pipeline` query parameter on hybrid searches.

### Indexing operations

`backend/app/opensearch/chunks.py`:

- `index_chunks()` — bulk-indexes chunks with embeddings; refreshes index once at end
- `delete_chunks_for_document()` — `delete_by_query` on `doc_id` before re-ingestion or on document delete
- `count_chunks_for_document()` — verifies delete completeness

---

## 5. Ingestion pipeline

**Orchestrator:** `backend/app/ingestion/pipeline.py` → `run_ingestion()`

`run_ingestion()` detects format from the uploaded file suffix and dispatches:

| Format | Extractor | Entry point |
|--------|-----------|-------------|
| `.pdf` | pdfplumber per-page | `extract_page_chunks()` in `text.py` |
| `.docx` | python-docx body-order | `extract_docx_chunks()` in `docx_extract.py` |

Shared downstream path for all formats: `embed_texts()` → `delete_chunks_for_document()` → `index_chunks()`.

**PDF per-page entry point:** `backend/app/ingestion/text.py` → `extract_page_chunks()`

Each page is processed in this order:

```
Page N
  ├── 1. Extract tables (+ record table bboxes)
  ├── 2. Extract text OUTSIDE table bboxes → paragraph split → word-chunk with overlap
  └── 3. Extract images/charts (excluding table regions)
```

Then globally:

```
all chunks → embed_texts([content...]) → index_chunks()
```

### Text chunking

**Location:** `backend/app/ingestion/chunking.py`

| Setting | Default |
|---------|---------|
| `CHUNK_MAX_WORDS` | 400 |
| `CHUNK_OVERLAP_WORDS` | 50 |

- Sliding word windows with overlap (`step = max_words - overlap_words`)
- Paragraphs shorter than 8 words are skipped (`MIN_TEXT_WORDS`)
- Words inside detected table bounding boxes are excluded from body text to avoid duplication
- **PDF text chunks** also store geometry for citation highlighting:
  - `bbox` — union of all words in the chunk (`[x0, top, x1, bottom]`, PDF top-origin)
  - `extra_metadata.line_bboxes` — one tight box per visual line (avoids shading column gutters on multi-column pages)

### DOCX ingestion (Phase 1)

**Location:** `backend/app/ingestion/docx_extract.py` → `extract_docx_chunks()`

| Aspect | Detail |
|--------|--------|
| **Library** | `python-docx` |
| **Method** | `extraction_method: "docx_native"` |
| **Scope** | Body-order **text** paragraphs and **tables** (no embedded images in Phase 1) |
| **Block order** | Walks `w:p` / `w:tbl` in document body XML (not separate `paragraphs` / `tables` lists) |
| **Pagination** | No fixed pages — `page_number` is a **1-based block ordinal** in reading order |
| **Section tracking** | Latest `Title` / `Heading*` style stored as `extra_metadata.section` |
| **Tables** | One table = one chunk; same markdown + `chart_profile` path as PDF tables |
| **Highlighting** | No `bbox` / `line_bboxes` (no PDF viewer for DOCX yet) |

**Metadata example:**

```json
{
  "source_format": "docx",
  "block_index": 12,
  "section": "Lists"
}
```

Citations in the UI show the **section name** (e.g. `Lists`) or `part N`, not `p. N`.

During ingestion, `update_document_record()` reports:

| Progress | Stage |
|----------|-------|
| 2% | Queued |
| 5–55% | Parsing pages (`Parsed page X/Y`) — PDF only |
| 30% | Parsing document — DOCX only |
| 65% | Generating embeddings |
| 85% | Indexing chunks |
| 100% | Completed or failed |

---

## 6. Modality handling: text, tables, images

### Text (`chunk_type: "text"`)

| Aspect | Detail |
|--------|--------|
| **Extractor** | pdfplumber word-level extraction |
| **Method** | `extraction_method: "pdfplumber"` |
| **Content** | Plain prose from page, excluding table regions |
| **Chunking** | Paragraphs → overlapping word windows (400 words, 50 overlap) |
| **At query time** | Included in LLM context |

**Flow:**

1. `page.extract_words()` filtered to words **outside** table bboxes
2. Words grouped into paragraphs (vertical gap heuristic), then overlapping word windows (mirrors `chunk_text()`)
3. Each window gets `bbox` (union) and `extra_metadata.line_bboxes` (per visual line via `_group_lines()`)

---

### Tables (`chunk_type: "table"`)

| Aspect | Detail |
|--------|--------|
| **Primary extractor** | pdfplumber `find_tables()` |
| **Fallback extractors** | Geometry reconstruction → text-line reconstruction → Camelot (lattice/stream) |
| **Content format** | Markdown table (`\| col \| col \|`) |
| **Chunking** | **One table = one chunk** (never split across rows) |
| **At query time** | Included in LLM context (LLM can render comparisons as markdown tables) |

**Quality gating** (`backend/app/ingestion/pdf_tables.py`):

Before accepting a pdfplumber table, the pipeline checks:

- **Column misalignment ratio** — fraction of data rows with wrong column count (threshold: 15%)
- **Semantic label loss** — numeric-heavy table with empty label column (common in scrambled PDF text layers)
- **Merged header geometry** — unusually wide cells suggesting merged headers

If quality checks fail → **recovery cascade:**

1. **Geometry reconstruction** (`table_geometry.py`) — cluster word bboxes into rows/columns inside table bbox
2. **Text-line fallback** — line-based parsing when geometry fails
3. **Camelot** — optional `lattice` then `stream` modes per page

Recovered tables are validated via `validate_reconstructed_table()` (label column density, column alignment, empty-column checks).

**Metadata stored in `extra_metadata`:**

```json
{
  "extraction_method": "pdfplumber | geometry | text_lines | camelot | camelot_optional",
  "table_headers": ["Revenue", "2024", "2025"],
  "chart_profile": {
    "chartable": true,
    "orientation": "wide",
    "period_count": 2,
    "metric_count": 6,
    "suggested_chart_type": "bar",
    "period_labels": ["2024", "2025"],
    "value_axis_label": "CNY Million",
    "period_column_indices": [2, 1]
  },
  "table_qa": {
    "misalignment_ratio": 0.0,
    "semantic_label_loss": false,
    "fallback_triggered": false,
    "recovery_method": "geometry",
    "recovery_validated": true
  }
}
```

`chart_profile` is set at ingestion when the table's shape reduces cleanly to metric-by-period series (see [§9](#9-computed-charts-from-table-chunks)). Tables without a valid profile omit this field.

---

### Images / charts (`chunk_type: "image"`)

| Aspect | Detail |
|--------|--------|
| **Approach** | OCR-proxy (no vision LLM in MVP) |
| **Method** | `extraction_method: "ocr_proxy"` |
| **Content** | Synthetic text: page context + nearby words + OCR output |
| **Storage** | PNG crops saved to `data/images/{user_id}/{doc_id}/page{N}_img{i}.png` |
| **At query time** | **Not** sent to LLM; `image_url` returned in sources for UI display |

**Two detection paths** (`backend/app/ingestion/images.py`):

#### 1. Embedded raster images

- From `page.images` (pdfplumber)
- Skip if area < 8,000 px² or overlaps a table bbox
- Crop at 150 DPI, save PNG

#### 2. Vector chart regions

- Cluster pdfplumber `rects`, `lines`, `curves` into chart-like regions
- Requires ≥ 20 vector objects, area ≥ 20,000 px², min 80×60 px
- Saved as `page{N}_vec{i}.png`

**Building retrievable content:**

```
Page {N} image/chart context.
Nearby text: {words within 80px margin of bbox, up to 80 words}
OCR text: {pytesseract output, max 500 chars}
```

**Skip decorative images** when both nearby text and OCR are empty.

**OCR dependency:** Requires `pytesseract` + Tesseract installed. If unavailable, OCR returns `""` silently; nearby text alone may still make the chunk indexable.

**Image serving:** Extracted images are served via authenticated `GET /images/{doc_id}/{filename}`. The backend verifies document ownership before reading from disk. The API returns `image_url` paths like `/images/{doc_id}/page12_img0.png`; the frontend loads them through `authFetch` + blob URLs (img tags cannot send Bearer headers).

---

### Modality comparison at a glance

| | Text | Table | Image |
|---|------|-------|-------|
| **Embedded field** | Prose | Markdown table | OCR + proximity text |
| **LLM context** | ✅ Yes | ✅ Yes | ❌ No (display only) |
| **BM25 searchable** | ✅ | ✅ | ✅ |
| **Vector searchable** | ✅ | ✅ | ✅ |
| **UI rendering** | Snippet | Snippet (markdown) + optional computed chart | Inline image + snippet |

---

## 7. Retrieval (hybrid search)

**Service:** `backend/app/retrieval/service.py` → `hybrid_retrieve()`

**Search:** `backend/app/opensearch/search.py` → `hybrid_search()`

### Algorithm

```python
query_vector = embed_texts([query])[0]

# OpenSearch hybrid query
{
  "size": k,
  "query": {
    "hybrid": {
      "queries": [
        {"match": {"content": query_text}},           # BM25
        {"knn": {"embedding": {"vector": query_vector, "k": k}}}  # vector
      ]
    }
  },
  "post_filter": {"bool": {"filter": [
    {"term": {"user_id": user_id}},
    {"term": {"doc_id": doc_id}}   // optional scope
  ]}}
}
```

Executed with `search_pipeline=hybrid-search-pipeline` for min-max normalization and 50/50 score fusion.

### Parameters

| Param | Default | Where set |
|-------|---------|-----------|
| `top_k` | 8 | **`DEFAULT_TOP_K` in `.env` only** — not exposed in the frontend UI |
| `doc_id` | `null` (search all **owned** docs) | Optional scope filter in sidebar or API |

All hybrid and k-NN queries **must** include a `user_id` filter derived from the authenticated JWT — never from the request body.

### Result mapping

Each hit becomes a `RetrievedChunk`:

- `chunk_id`, `doc_id`, `filename`, `page_number`, `chunk_type`, `content`
- `score` — fused hybrid score
- `bbox` — chunk-level union box (PDF text/tables/images)
- `image_url` — mapped from `image_path` for image chunks
- `extraction_method` — from `extra_metadata`
- `extra_metadata` — includes `chart_profile`, `line_bboxes` (PDF text), `source_format`, `section` (DOCX)

The `/query/stream` **sources** SSE payload adds viewer fields: `doc_id`, `bbox`, `line_bboxes`, `page_count`, `source_format`, `section` (see [§8](#8-query-and-answer-generation) and [§19](#19-pdf-citation-viewer-and-highlighting)).

### Request logging

**Location:** `backend/app/retrieval/request_log.py`

Every `/query/stream` and `/search` request logs:

1. **`RETRIEVAL_REQUEST`** — compact line: retrieved count, table count, chartable marked, charts offered, chunk type breakdown
2. **`RETRIEVAL_REQUEST_DETAIL`** — full JSON: per-chunk summary + `chart_eligibility` (`validation_outcome`: `offered`, `validation_failed`, `not_marked_at_ingestion`, etc.)

Query stream completion logs `QUERY_STREAM_DONE ok=true/false`.

### Standalone search API

`GET/POST /search` exposes retrieval without LLM generation — useful for debugging retrieval quality.

---

## 8. Query and answer generation

**Locations:** `backend/app/api/query.py`, `backend/app/llm/groq.py`, `backend/app/llm/agent.py`, `backend/app/llm/tools.py`

### Query paths overview

| Mode | Env flag | Retrieval | Image handling | Answer LLM |
|------|----------|-----------|----------------|------------|
| **Legacy** | `AGENT_ENABLED=false` (default) | Always `hybrid_retrieve` on every query | `classify_visual_intent` + proximity attach | `stream_groq_answer(query, context)` |
| **Agent** | `AGENT_ENABLED=true` | Router calls tools when needed (multi-round) | `search_images` tool + proximity attach | Same grounded stream — **not** router text |

Enable agent mode in `.env`:

```bash
AGENT_ENABLED=true
AGENT_MODEL=llama-3.3-70b-versatile
AGENT_MAX_ROUNDS=3
AGENT_HISTORY_TURNS=6
AGENT_HISTORY_MAX_CHARS=1500
```

### Legacy path (`AGENT_ENABLED=false`)

**Location:** `_legacy_event_stream()` in `query.py`

1. Embed query → `hybrid_retrieve()` (always).
2. In parallel, `classify_visual_intent()` — if `visual_intent=required`, run `retrieve_intent_images()`.
3. `_assemble_retrieval_payload()` — context, sources, charts, hero images, optional `visual_note`.
4. Stream answer via `stream_groq_answer()`.

### Agent path (`AGENT_ENABLED=true`)

**Location:** `iter_agent_turn()` in `llm/agent.py`, `_agent_event_stream()` in `query.py`

```
User message + chat history
        │
        ▼
Groq router (tool-calling, up to AGENT_MAX_ROUNDS)
        │
        ├── Greeting / thanks ──────────────► direct reply (no search)
        ├── Ambiguous scope ────────────────► clarification question (no search)
        ├── Document question ────────────► search_documents (required)
        ├── Explicit visual request ──────► search_images (optional + search_documents)
        └── "What files do I have?" ──────► list_documents → maybe search_documents
        │
        ▼ (after tools or stop)
Merged chunks (deduped by chunk_id, best score wins)
        │
        ▼
stream_groq_answer(query, context)   ← excerpts only; no chat history here
```

#### Agent tools

| Tool | Implementation | Returns |
|------|----------------|---------|
| `search_documents` | `hybrid_retrieve()` | Text/table/image-metadata chunks |
| `search_images` | `retrieve_intent_images()` | Top image chunk(s) for explicit visual queries |
| `list_documents` | `list_document_records()` | User's files, status, chunk counts |

Tool schemas: `AGENT_TOOLS` in `llm/agent.py`. Executors: `llm/tools.py`.

#### Multi-round routing

The router may call tools across **multiple Groq rounds** (default max **3**, `AGENT_MAX_ROUNDS`):

- Round 1: `list_documents` when scope is unclear.
- Round 2: `search_documents` with `doc_id` and refined query.
- Round 3: Broader re-search if first pass returned weak hits.

Chunks from all rounds are **merged** (`_merge_retrieved_chunks`) before the answer step.

#### Follow-ups and query rewriting

- **Router prompt** instructs rewriting follow-ups into standalone queries (resolve pronouns from chat history).
- **Forced-search fallback** — if the router answers a document question in plain text without tools, the backend automatically runs `search_documents`.
- **`rewrite_retrieval_query()`** — when fallback fires on pronoun follow-ups ("show an image of **him**"), a small Groq call rewrites using prior turns before retrieval.

Chat history is sent to the **router only** (`AGENT_HISTORY_TURNS`, truncated per message to `AGENT_HISTORY_MAX_CHARS`). The **answer** LLM never sees prior turns — only retrieved excerpts — to prevent hallucination from world knowledge or stale chat context.

#### Hallucination guards

| Guard | Behavior |
|-------|----------|
| Router prompt | Must not invent document facts; must call tools for factual Q&A |
| Forced search | Non-greeting document questions without tool calls → `search_documents` |
| Answer prompt | Never use general world knowledge; only excerpts; missing fact → `"Not found in the provided documents"` |
| Grounding path | Agent answers always use `stream_groq_answer()`, not router direct text (except greetings/clarification) |

#### SSE persistence note

The agent loop runs **inside** the SSE generator (after the HTTP handler returns). ORM objects from the request scope are detached, so:

- `user_id` / `chat_id` are captured as integers before streaming.
- Assistant messages are saved via `_persist_assistant_reply()` using a **fresh** `SessionLocal()` session.

### Context construction

Only **text** and **table** chunks are passed to the answer LLM:

```
--- Source 1 ---
Document: huawei_2025.pdf
Page: 12
Type: table
Content:
| Metric | 2024 | 2025 |
| --- | --- | --- |
| Revenue | 100 | 120 |
```

For DOCX sources, `Page:` in the LLM context carries the block ordinal; the UI shows `section` instead of page numbers in citations.

Image chunks in top-k still appear in the **sources** payload with `image_url` for the frontend `SourcesPanel` and hero strip.

### LLM configuration

| Setting | Value |
|---------|-------|
| Provider | Groq Cloud API |
| Answer model | `llama-3.3-70b-versatile` |
| Agent router model | `AGENT_MODEL` (default `llama-3.3-70b-versatile`) |
| Legacy visual intent model | `IMAGE_INTENT_MODEL` (default `llama-3.1-8b-instant`) |
| Temperature | 0.1 |
| Streaming | SSE token events |

### Grounding rules (system prompt)

**Location:** `SYSTEM_PROMPT` in `llm/groq.py`

The assistant is a **domain-agnostic document Q&A** agent (not tied to financial filings). Rules include:

- Answer **only** from provided excerpts
- If fact missing → `"Not found in the provided documents"`
- Never invent numbers, dates, names, or entities; **never** use general world knowledge about companies or people
- **No inline citations** in the answer — the Sources panel handles citations separately
- Use markdown **block** tables for multi-category numeric comparisons; **never** embed table pipe syntax (`| ... |`) inside bullets or sentences (logged as `LLM_ANSWER_INLINE_TABLE` when detected)
- Keep single-fact answers concise

### SSE event stream

| Event | Payload | Notes |
|-------|---------|-------|
| `meta` | `{ query, top_k, doc_id, session_id, agent }` | Sent first; `agent: true` when agent mode on |
| `tool` | `{ name, status: "running" \| "complete", round? }` | Agent mode only; live tool progress |
| `token` | `{ token: "..." }` | Repeated per delta |
| `sources` | `{ sources: [{ chunk_id, doc_id, filename, page_number, chunk_type, snippet, image_url, score, source_format, section, bbox, line_bboxes, page_count, attached_images, attach_reason }] }` | Always emitted (may be empty) |
| `charts` | `{ charts: [...] }` | Only when chartable table chunks retrieved |
| `done` | `{ ok: true/false }` | |
| `error` | `{ message: "..." }` | |

### Image attachment and hero strip

**Location:** `backend/app/retrieval/image_attach.py`, `frontend/src/lib/heroImages.ts`, `frontend/src/components/HeroImages.tsx`

| Track | When | Mechanism |
|-------|------|-----------|
| **Intent (A)** | User explicitly asks to see a photo/chart | Legacy: `classify_visual_intent`; Agent: `search_images` tool |
| **Proximity (B)** | Implicit relevance (e.g. "who is the chairman") | Attach images near top text/table hits by bbox proximity |

Post-retrieval enrichment in `_assemble_retrieval_payload()`:

- `resolve_proximity_attachments()` — bbox/column overlap scoring on PDF pages.
- `build_display_images()` — dedup by IoU, cap display count.
- **Hero cap:** 1 image when `search_images` / visual intent required; else `IMAGE_MAX_DISPLAY` (default 2).
- **`visual_note`** — short UI hint to the answer LLM when an image is shown separately (prevents "not found" when hero image is visible).

Image-only agent results (no text chunks) still resolve `page_count` on sources from the document registry so the PDF viewer can open correctly.

### Frontend (Chat UI)

| Feature | Behavior |
|---------|----------|
| **Auth** | `/login` and `/signup` pages; access token in memory; refresh via httpOnly cookie |
| **Layout** | Compact header (title dropdown with doc/chunk counts, logout top-right); collapsible icon sidebar (New chat, Chat, Documents) |
| **Scope** | Optional `doc_id` filter in sidebar (indexed docs for **current user** only) |
| **Conversations** | Sidebar lists named chat sessions; select to reload history; "New chat" creates empty session |
| **Library sidebar** | Shows **latest 3** uploaded documents (newest first) |
| **Agent status** | While tools run, placeholder shows "Searching documents…" / "Finding images…" |
| **top_k** | Not configurable in UI — backend `DEFAULT_TOP_K` only (router may pass different `top_k` in tool args) |
| **Sources panel** | Expandable snippets; compact image thumbnails in expanded rows; PDF `p. N`, DOCX **section** or `part N` |
| **PDF viewer** | "Open page N in document" — side panel with range-streamed PDF.js; uses `pdf.numPages` when source `page_count` is missing |
| **DOCX sources** | "Open in document" switches to Documents tab (no PDF viewer yet) |
| **Hero images** | Strip above sources for top attached/intent images (capped) |
| **Computed charts** | SVG bar/line below answer; distinct colors; labeled "Computed chart · derived from table" |

---

## 9. Computed charts (from table chunks)

Computed charts are a **second visualization layer**, distinct from PDF chart images in Sources. They plot numeric values parsed from retrieved **table** chunks.

**Location:** `backend/app/charts/` (profile, columns, period, units, spec, service) · `frontend/src/components/ComputedChartsPanel.tsx`

### When a chart appears

Charts are **not** triggered by query wording. No keyword matching, no LLM chart generation.

```
Ingestion:  table shape → chart_profile (chartable: true/false)
Query:      hybrid search retrieves chunk → re-validate → emit chart spec
```

Both must be true:

1. Table was marked **`chart_profile.chartable`** at ingestion.
2. That table chunk appears in **top-k retrieval** for the query.

If either fails, no computed chart (fail closed).

### Chart types

| Type | Structural rule | UI |
|------|-----------------|-----|
| **Bar / grouped bar** | 1–8 metrics, 2–5 periods | Colored grouped bars |
| **Line** | 1 metric, 3–5 periods | Single line + points |
| **(none)** | Validation fails or shape invalid | LLM may still answer using table markdown in prose |

Out of scope: pie, scatter, dual-axis, maps.

### Ingestion-time analysis

**`analyze_table_chartability()`** (`profile.py`) uses only structural properties:

- **Wide layout** — metrics in rows, periods in column headers (or embedded years like `(CNY Million) 2025`)
- **Long layout** — periods in row labels, metrics in column headers
- **Annotation columns** excluded structurally — short acronym headers (≤3 chars), or columns dominated by `%` values (e.g. YoY)
- **Duplicate period keys** deduped when all metric values agree; rejected when conflicting
- **Composition rows** (values summing to ~100%) rejected as pie-like data

Stored in `extra_metadata.chart_profile`:

```json
{
  "chartable": true,
  "orientation": "wide",
  "period_count": 2,
  "metric_count": 6,
  "suggested_chart_type": "bar",
  "period_labels": ["2024", "2025"],
  "value_axis_label": "CNY Million"
}
```

### Y-axis unit detection

**`detect_value_axis_label()`** (`units.py`) reads parenthetical unit markers from header cells:

- `(CNY Million)`, `(USD Million)`, `%`, scale words (million, billion, …)
- Fails closed to **`"Value"`** when markers are missing or contradictory (e.g. mixed currencies)
- No hardcoded company or metric names — pattern-based only

### Query-time validation

**`validate_and_build_chart_spec()`** (`spec.py`):

1. Re-parses table markdown from chunk `content`
2. Re-runs chartability analysis; must match stored profile counts/orientation
3. Builds `{ periods, series, value_axis_label, period_axis_label: "Period" }`
4. Numbers come **only** from parsed cells — never from the LLM

### PDF image vs computed chart

| | PDF image chunk | Computed chart |
|---|-----------------|----------------|
| **Source** | Raster/vector crop from PDF | Parsed table numbers |
| **Trigger** | Image chunk in top-k | Chartable table chunk in top-k |
| **LLM input** | No | No |
| **UI location** | Sources panel (expand row) | Below answer, labeled **derived** |
| **Priority** | Primary visual when both match | Secondary (`is_secondary: true`) when an image chunk is also retrieved |

### Example queries (Huawei segment table, p.23)

These retrieve the chartable table chunk (after re-ingest):

- `ICT Infrastructure revenue 2025 2024`
- `Huawei segment revenue China EMEA`

These typically **do not** produce computed charts (retrieve text/image only):

- `Show me Huawei's financial highlights` — image chunks in Sources, no table in top-k
- `Compare Timberland net interest margin` — no chartable Timberland tables indexed

**Re-ingest required** after chartability or unit-detection code changes so chunks get fresh `chart_profile`.

---

## 10. Authentication, signup, and sessions

**Locations:** `backend/app/api/auth.py`, `backend/app/auth/`, `frontend/src/pages/LoginPage.tsx`, `frontend/src/pages/SignupPage.tsx`, `frontend/src/auth/`

### Overview

Users sign up with **email + password** (bcrypt, cost factor 12). The backend issues:

| Token | Transport | TTL | Purpose |
|-------|-----------|-----|---------|
| **Access JWT** | `Authorization: Bearer` header | ~15 min (`ACCESS_TOKEN_TTL_MINUTES`) | Authenticates API requests; `sub` = `user_id` |
| **Refresh token** | httpOnly cookie (`rag_refresh`, path `/auth` only) | ~7 days (`REFRESH_TOKEN_TTL_DAYS`) | Silent session renewal; hashed (SHA-256) in PostgreSQL |

The refresh cookie is **not** sent to `/documents`, `/query`, or `/images` — only to `/auth/refresh` and `/auth/logout`.

### Auth API

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `POST` | `/auth/signup` | — | Create account; returns access token + sets refresh cookie |
| `POST` | `/auth/login` | — | Verify password; returns access token + sets refresh cookie |
| `POST` | `/auth/refresh` | Cookie | Rotate refresh token; issue new access JWT |
| `POST` | `/auth/logout` | Cookie | Revoke refresh token; clear cookie |
| `GET` | `/auth/me` | Bearer | Return current user `{ id, email }` |
| `POST` | `/auth/change-password` | Bearer | Change password; revoke **all** refresh sessions |

### Frontend auth flow

1. **Signup / login** — access token stored **in memory** (`tokenStore.ts`); refresh cookie set automatically.
2. **`authFetch`** — attaches Bearer token; on 401, calls `/auth/refresh` once, retries, or redirects to `/login`.
3. **Route guard** — `RequireAuth` wraps the main app at `/`; unauthenticated users see `/login` or `/signup` only.

### Login / signup UI

| Page | Layout | Left column | Right column |
|------|--------|-------------|--------------|
| **Login** | Two columns (desktop) | Animated typewriter headline cycling RAG value phrases | Email + password form |
| **Signup** | Two columns (desktop) | Registration form | Auto-rotating feature carousel (grounded answers, citations, charts, privacy) |

Both pages share a unified dark gradient background that fades across the column seam. On mobile, only the form column is shown.

### Refresh rotation

Each `/auth/refresh` call:

1. Validates the presented refresh token (not revoked, not expired).
2. Sets `revoked_at` on the old token row.
3. Issues a new opaque refresh token and sets a new cookie.

Replaying a previously rotated token returns **401**.

### PostgreSQL tables (auth)

| Table | Key fields |
|-------|------------|
| `users` | `id`, `email` (unique), `password_hash`, `created_at` |
| `refresh_tokens` | `id`, `user_id`, `token_hash`, `expires_at`, `revoked_at`, `created_at` |

Email is normalized to lowercase on signup. Uniqueness is enforced at the DB level (`UNIQUE` on `email`).

---

## 11. Per-user document isolation

Every read and write is scoped by `user_id` from the JWT — **never** from the request body.

### OpenSearch

Both `rag_documents` and `rag_chunks` include a `user_id` keyword field. Hybrid search always filters:

```json
{"term": {"user_id": "<current_user_id>"}}
```

### Filesystem

| Resource | Path pattern |
|----------|--------------|
| Uploads | `data/uploads/{user_id}/{doc_id}/{filename}` |
| Images | `data/images/{user_id}/{doc_id}/page{N}_img{i}.png` |

### Ownership checks

| Action | Behavior when `doc_id` belongs to another user |
|--------|-----------------------------------------------|
| `GET /documents/{doc_id}/status` | **404** |
| `DELETE /documents/{doc_id}` | **404** |
| `GET/POST /search?doc_id=...` | **404** |
| `POST /query/stream` with `doc_id` | **404** |
| `GET /images/{doc_id}/{filename}` | **404** |

Returning 404 (not 403) avoids leaking whether a document ID exists.

### Migration / legacy data

On startup, `ensure_indices()` deletes OpenSearch indices that lack a `user_id` mapping and wipes `data/uploads` + `data/images`. Users re-upload documents after deploy — no migration script.

### Frontend

- Document list, scope dropdown, upload, and delete all use `authFetch`.
- Each user sees only their own indexed documents in the sidebar library.

---

## 12. Chat history

**Locations:** `backend/app/api/chats.py`, `backend/app/chat/`, `frontend/src/api/chats.ts`

### PostgreSQL tables

| Table | Key fields |
|-------|------------|
| `chat_sessions` | `id`, `user_id`, `title`, `created_at`, `updated_at` |
| `chat_messages` | `id`, `session_id`, `role` (`user` \| `assistant`), `content`, `sources` (JSON), `charts` (JSON), `created_at` |

### Chat API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/chats` | List sessions for current user (newest `updated_at` first) |
| `POST` | `/chats` | Create empty session (title: "New chat") |
| `GET` | `/chats/{session_id}` | Load messages (404 if not owned) |
| `DELETE` | `/chats/{session_id}` | Delete session + all messages |

### Persistence during query

`POST /query/stream` accepts optional `session_id`:

1. If `session_id` provided → verify ownership; **404** if missing or not owned.
2. If omitted → create new session; title auto-set from first user message (truncated to 60 chars).
3. Append **user message** before streaming starts.
4. After successful stream (`done: ok`) → append **assistant message** with full answer, `sources`, and `charts` via `_persist_assistant_reply()` (fresh DB session — required because SSE continues after the request-scoped session closes).
5. Return `session_id` in the `meta` SSE event so the frontend can track the active conversation.

**Note:** Chat history is persisted for UI reload and sent to the **agent router** when `AGENT_ENABLED=true` (`history_for_llm()`, last N turns). The **answer** LLM remains single-turn (retrieved excerpts only) to keep answers grounded. Legacy mode does not send history to Groq.

### Frontend

- Sidebar **Conversations** list with select, delete, and **+ New chat**.
- Selecting a session loads messages (including sources and charts) from `GET /chats/{id}`.
- Sending a message passes `session_id`; list refreshes after stream completes.

---

## 13. Security hardening

Production-oriented controls added in Phase 4:

| Control | Implementation |
|---------|----------------|
| **Rate limiting** | Per-`user_id` sliding window (in-process): upload default 10/min, query default 30/min → **429** |
| **Refresh rotation** | Old refresh token revoked on each `/auth/refresh` |
| **Password change** | `POST /auth/change-password` revokes all refresh tokens for the user |
| **Email uniqueness** | DB `UNIQUE` constraint + signup race handling |
| **CORS** | Origins from `CORS_ORIGINS` env (comma-separated); `allow_credentials=True` |
| **JWT secret** | `REQUIRE_SECURE_JWT_SECRET=true` refuses startup if secret is missing or a known insecure default |
| **Isolation tests** | Cross-user 404 on documents, search, chats, and images |

Rate limits are per-process (fine for single-container deploy). No Redis — refresh revocation uses PostgreSQL `revoked_at`.

### Production checklist

```bash
JWT_SECRET=<long-random-string>
REQUIRE_SECURE_JWT_SECRET=true
COOKIE_SECURE=true
CORS_ORIGINS=https://your-frontend-domain.com
```

---

## 14. API reference

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `GET` | `/health` | — | Service health |
| `GET` | `/config` | — | Public viewer config (`pdf_viewer_page_window`) |
| `POST` | `/auth/signup` | — | Create account |
| `POST` | `/auth/login` | — | Sign in |
| `POST` | `/auth/refresh` | Cookie | Rotate refresh token |
| `POST` | `/auth/logout` | Cookie | Revoke refresh token |
| `GET` | `/auth/me` | Bearer | Current user |
| `POST` | `/auth/change-password` | Bearer | Change password; revoke all sessions |
| `POST` | `/documents/upload` | Bearer | Upload **PDF or DOCX**, start background ingestion |
| `GET` | `/documents` | Bearer | List **current user's** documents |
| `GET` | `/documents/{doc_id}/status` | Bearer | Poll ingestion progress (owned only); includes `page_count` for PDFs |
| `GET` | `/documents/{doc_id}/file` | Bearer | Serve original PDF with **HTTP Range** (206) for PDF.js streaming |
| `DELETE` | `/documents/{doc_id}` | Bearer | Delete document, chunks, and files (owned only) |
| `GET/POST` | `/search` | Bearer | Hybrid retrieval only (no LLM); user-scoped |
| `POST` | `/query/stream` | Bearer | Full RAG: retrieve + stream answer + sources + persist chat |
| `GET` | `/chats` | Bearer | List chat sessions |
| `POST` | `/chats` | Bearer | Create empty chat session |
| `GET` | `/chats/{session_id}` | Bearer | Load chat messages |
| `DELETE` | `/chats/{session_id}` | Bearer | Delete chat session |
| `GET` | `/images/{doc_id}/{filename}` | Bearer | Serve extracted chart images (owned only) |

Interactive docs: `http://localhost:8000/docs`

---

## 15. Configuration

All settings in `backend/app/config.py`, loaded from `.env`:

```bash
# OpenSearch
OPENSEARCH_HOST=opensearch      # use localhost when running backend outside Docker
OPENSEARCH_PORT=9200

# LLM (required for /query/stream)
GROQ_API_KEY=your_groq_api_key_here

# Embeddings (do not change without reindexing)
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Storage
IMAGES_DIR=data/images
UPLOADS_DIR=data/uploads

# Text chunking
CHUNK_MAX_WORDS=400
CHUNK_OVERLAP_WORDS=50

# Retrieval (backend only — not exposed in frontend UI)
DEFAULT_TOP_K=8

# Image attachment (PDF only)
IMAGE_ATTACH_ENABLED=true
IMAGE_MAX_DISPLAY=2
IMAGE_INTENT_ENABLED=true
IMAGE_INTENT_MODEL=llama-3.1-8b-instant
IMAGE_INTENT_TOP_K=3

# Agent router (Groq tool-calling; replaces always-on retrieval when true)
AGENT_ENABLED=false
AGENT_MODEL=llama-3.3-70b-versatile
AGENT_MAX_ROUNDS=3
AGENT_HISTORY_TURNS=6
AGENT_HISTORY_MAX_CHARS=1500

# PDF citation viewer (windowed page render: pages above/below viewport)
PDF_VIEWER_PAGE_WINDOW=2

# Auth + PostgreSQL
DATABASE_URL=postgresql+psycopg2://rag:rag@postgres:5432/rag
JWT_SECRET=generate-a-long-random-secret-for-production
ACCESS_TOKEN_TTL_MINUTES=15
REFRESH_TOKEN_TTL_DAYS=7
REFRESH_COOKIE_NAME=rag_refresh
COOKIE_SECURE=false
COOKIE_SAMESITE=lax

# Security hardening
REQUIRE_SECURE_JWT_SECRET=false
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
UPLOAD_RATE_LIMIT_PER_MINUTE=10
QUERY_RATE_LIMIT_PER_MINUTE=30
```

Change `DEFAULT_TOP_K` in `.env` and restart the backend to tune how many chunks are retrieved per query (legacy path and agent tool default).

| Agent setting | Purpose |
|---------------|---------|
| `AGENT_ENABLED` | `true` = tool-calling router; `false` = legacy always-on hybrid search |
| `AGENT_MAX_ROUNDS` | Max Groq router ↔ tool loops per user message (default 3) |
| `AGENT_HISTORY_TURNS` | Prior user/assistant pairs sent to router |
| `AGENT_HISTORY_MAX_CHARS` | Per-message truncation for router (avoids Groq 400 on long replies) |

**Never commit `.env`** — it is gitignored. Use `.env.example` as the template.

### OpenSearch index names (code defaults)

| Index | Name |
|-------|------|
| Chunks | `rag_chunks` |
| Documents | `rag_documents` |
| Search pipeline | `hybrid-search-pipeline` |

---

## 16. Project layout

```
multimodal-rag/
├── docker-compose.yml          # OpenSearch, PostgreSQL, backend, frontend
├── .env.example
├── ARCHITECTURE.md             # Design reference
├── RAG_GUIDE.md                # This document
├── data/
│   ├── images/{user_id}/{doc_id}/   # Extracted chart/image PNGs
│   └── uploads/{user_id}/{doc_id}/  # Uploaded PDFs and DOCX files
├── backend/
│   └── app/
│       ├── main.py             # FastAPI app, CORS, lifespan
│       ├── config.py           # Settings
│       ├── db/
│       │   ├── models.py       # User, RefreshToken, ChatSession, ChatMessage
│       │   └── session.py      # SQLAlchemy session + init_db()
│       ├── auth/
│       │   ├── service.py      # Users, refresh tokens, password change
│       │   ├── security.py     # bcrypt, JWT, refresh token hashing
│       │   ├── dependencies.py # get_current_user
│       │   ├── rate_limit.py   # Per-user upload/query rate limits
│       │   └── schemas.py      # SignupRequest, TokenResponse, etc.
│       ├── chat/
│       │   ├── service.py      # Session CRUD, message persistence
│       │   └── schemas.py      # ChatSessionDetail, etc.
│       ├── api/
│       │   ├── auth.py         # /auth/* endpoints
│       │   ├── chats.py        # /chats/* endpoints
│       │   ├── documents.py    # Upload (PDF/DOCX), list, delete (user-scoped)
│       │   ├── pdfs.py         # GET /documents/{doc_id}/file (range serving)
│       │   ├── health.py         # /health, /config
│       │   ├── query.py        # SSE RAG endpoint + enriched sources payload
│       │   ├── search.py       # Retrieval-only endpoint (user-scoped)
│       │   └── images.py       # Authenticated image serving
│       ├── ingestion/
│       │   ├── pipeline.py     # Orchestration + format dispatch (user-scoped paths)
│       │   ├── text.py         # PDF page extraction + text bboxes / line_bboxes
│       │   ├── docx_extract.py # DOCX body-order text + tables
│       │   ├── chunking.py     # Word overlap chunking
│       │   ├── tables.py       # Markdown + validation helpers
│       │   ├── pdf_tables.py   # Table extraction + fallbacks
│       │   ├── table_geometry.py  # Coordinate-based reconstruction
│       │   ├── images.py       # Image/chart OCR-proxy chunks
│       │   ├── embeddings.py   # FastEmbed + L2 normalize
│       │   └── models.py       # ExtractedChunk dataclass
│       ├── opensearch/
│       │   ├── client.py       # OpenSearch client factory
│       │   ├── bootstrap.py    # Startup: indices + pipeline
│       │   ├── indices.py      # Index mappings (incl. user_id)
│       │   ├── pipelines.py    # Hybrid search pipeline
│       │   ├── chunks.py       # Index/delete chunks
│       │   ├── documents.py    # Document registry CRUD (user-scoped)
│       │   └── search.py       # knn_search, hybrid_search (user_id filter)
│       ├── charts/
│       │   ├── profile.py      # Ingestion chartability analysis
│       │   ├── columns.py      # Period/metric/annotation column roles
│       │   ├── period.py       # Period label + embedded year detection
│       │   ├── units.py        # Y-axis label from header parentheticals
│       │   ├── spec.py         # Query-time validation + chart spec
│       │   └── service.py      # build_computed_charts()
│       ├── retrieval/
│       │   ├── service.py      # hybrid_retrieve(user_id=...)
│       │   ├── image_attach.py # Proximity attach + intent image helpers
│       │   ├── request_log.py  # RETRIEVAL_REQUEST, LLM_CONTEXT, LLM_ANSWER logging
│       │   └── models.py       # RetrievedChunk, SearchResponse
│       └── llm/
│           ├── groq.py         # Grounded answer streaming + SYSTEM_PROMPT
│           ├── agent.py        # Multi-round Groq router (iter_agent_turn)
│           ├── tools.py        # search_documents, search_images, list_documents
│           └── intent.py       # Visual intent classifier (legacy path only)
└── frontend/
    └── src/
        ├── main.tsx            # Router: /login, /signup, / (guarded)
        ├── App.tsx             # Chat + docs; conversations; scope filter
        ├── auth/
        │   ├── AuthContext.tsx # login, signup, logout, refresh on load
        │   ├── tokenStore.ts   # In-memory access token
        │   └── RequireAuth.tsx # Route guard
        ├── api/
        │   ├── auth.ts         # /auth/* client
        │   ├── chats.ts        # /chats/* client
        │   ├── client.ts     # Documents API + getViewerConfig()
        │   ├── query.ts        # SSE stream (authFetch)
        │   └── http.ts         # authFetch wrapper
        ├── pdf/
        │   ├── pdfjs.ts        # PDF.js worker + range loading task
        │   └── log.ts          # [pdf-viewer] prefixed console logs
        ├── pages/
        │   ├── LoginPage.tsx   # Two-column login + typewriter
        │   ├── SignupPage.tsx  # Two-column signup + feature carousel
        │   └── AuthForm.tsx    # Shared email/password form panel
        ├── lib/
        │   └── heroImages.ts         # Hero image dedup/cap (shared with history reload)
        └── components/
            ├── Typewriter.tsx          # Animated headline phrases
            ├── FeatureCarousel.tsx     # Signup feature cards
            ├── AuthImage.tsx           # Authenticated image blob loader
            ├── HeroImages.tsx          # Hero strip above sources
            ├── ComputedChartsPanel.tsx # SVG bar/line charts
            ├── SourcesPanel.tsx        # Citations + open-in-document actions
            ├── PdfViewerPanel.tsx    # Side-panel PDF.js viewer + highlights
            └── PdfViewerBoundary.tsx # Error boundary (viewer crash isolation)
```

---

## 17. Running locally

### With Docker Compose (recommended)

```bash
cp .env.example .env
# Set GROQ_API_KEY in .env

docker compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| OpenSearch | http://localhost:9200 |
| OpenSearch Dashboards | http://localhost:5601 |
| PostgreSQL | localhost:5432 (`rag` / `rag` / `rag`) |

### Manual flow to verify RAG

1. **Sign up** at http://localhost:5173/signup (or log in at `/login`).
2. Upload a **PDF or DOCX** in the Documents tab (or `POST /documents/upload` with Bearer token).
3. Poll until `ingestion_status` is `indexed`.
4. Ask a question in Chat (or `POST /query/stream` with Bearer token).
5. With `AGENT_ENABLED=true`, watch for tool status ("Searching documents…") then streamed answer + sources.
6. Confirm the conversation appears in the sidebar **Conversations** list.
7. Log out, log back in — history and documents should reload for the same user.
8. Sign up as a second user — should see empty library and no access to the first user's docs.
9. Inspect retrieved sources — expand snippets; image sources show compact thumbnails when expanded.
10. For **PDF citation highlights**: click "Open page N in document" on a text source — side panel should scroll to the page with amber (primary) and sky (secondary) marker overlays on cited lines.
11. For computed charts: re-ingest PDFs, then query content that matches a chartable table (see [§9](#9-computed-charts-from-table-chunks)).

### Test retrieval without LLM

```bash
# Obtain access token via signup/login, then:
curl -H "Authorization: Bearer <token>" \
  "http://localhost:8000/search?query=revenue+2025"
```

`top_k` defaults to `DEFAULT_TOP_K` from `.env` when omitted.

---

## 18. Tuning and debugging

### Retrieval quality

| Knob | Where to change |
|------|-----------------|
| BM25 vs vector balance | `weights` in `pipelines.py` (default 0.5/0.5) |
| Result count | `DEFAULT_TOP_K` in `.env` (restart backend) |
| Document scope | `doc_id` filter in sidebar or API |
| Text granularity | `CHUNK_MAX_WORDS`, `CHUNK_OVERLAP_WORDS` |

### Agent mode (`AGENT_ENABLED=true`)

Grep backend logs for `AGENT`:

```
AGENT turn_start user_id=2 history_turns=3 query_preview='...' max_rounds=3
AGENT groq_request model=llama-3.3-70b-versatile message_count=8 tools=3
AGENT tool_start round=1 name=search_documents args={"query": "..."}
AGENT tool_done round=1 name=search_documents chunks=8 images=0 total_chunks=8
AGENT chunks_retrieved tools=['search_documents'] total=8 by_type={'text': 6, 'image': 1, 'table': 1}
AGENT chunk[1] id=... type=text file=huawei.pdf page=13 score=0.85 chars=374 preview='...'
AGENT answer_stream mode=grounded context_chars=7022 sources=8
```

| Log signal | Likely cause |
|------------|--------------|
| `AGENT route_fallback reason=router_skipped_tools` | Router tried to answer without search — forced `search_documents` ran |
| `AGENT query_rewrite original=... rewritten=...` | Pronoun follow-up rewritten before retrieval |
| `AGENT route_clarification` | Router asked which document to use |
| `AGENT groq_request_failed status=400` | History too long — lower `AGENT_HISTORY_MAX_CHARS` or `AGENT_HISTORY_TURNS` |
| `DetachedInstanceError` on `current_user` | Stale backend image — rebuild; ensure `user_id` captured before SSE |
| `UnboundLocalError: user_id` | Same — pull latest `query.py` |

### LLM answer quality

Check backend logs for each `/query/stream` request:

```
LLM_CONTEXT query_preview='...' sources=8 context_chars=12450
LLM_CONTEXT_DETAIL {"query": "...", "context": "--- Source 1 ---\n..."}
LLM_ANSWER query_preview='...' answer_chars=820 inline_table_detected=false
LLM_ANSWER_DETAIL {"query": "...", "answer": "...", "inline_table_detected": false}
```

| Log signal | Likely cause |
|------------|--------------|
| `inline_table_detected=true` | Model jammed `\| --- \|` into a bullet/sentence — won't render as a table; tighten prompt or re-ask |
| `LLM_ANSWER_INLINE_TABLE` warning | Same issue, explicit guard fired |
| Short `context_chars` | Retrieval missed the right chunk — tune `DEFAULT_TOP_K` or scope `doc_id` |

### PDF citation viewer

Browser console uses `[pdf-viewer]` prefixed logs (`frontend/src/pdf/log.ts`):

```
[pdf-viewer] document.load.start { docId: "..." }
[pdf-viewer] document.load.ok { docId: "...", numPages: 340 }
[pdf-viewer] render.window { start: 12, end: 16, pageWindow: 2, rendered: 5 }
[pdf-viewer] page.render.ok { pageNumber: 14, scale: 1 }
```

| Symptom | Check |
|---------|-------|
| Viewer opens but **black screen** | Source `page_count` was 0 (common for image-only agent hits) — fixed: viewer uses `pdf.numPages`; re-query for updated sources |
| Viewer opens but **no highlights** | Chunk may predate bbox ingestion — **re-upload** the PDF |
| `Setting up fake worker failed` | Ensure `pdfjs.ts` uses `?worker` + `workerPort` (not `?url` alone) |
| Whole app blanks on second open | `PdfViewerBoundary` should catch viewer errors; doc load effect must depend on `docId` only |
| Slow open on late pages | Confirm `GET /documents/{doc_id}/file` returns **206** ranges and PDF.js has `disableAutoFetch: true` |

### Computed chart issues

Check backend logs for each query:

```
RETRIEVAL_REQUEST ... table=0 chartable_marked=0 charts_offered=0
```

| Log signal | Likely cause |
|------------|--------------|
| `table=0` | No table chunk in top-k — rephrase or scope to the right doc |
| `chartable_marked=0` | Table retrieved but no `chart_profile` — re-ingest PDF |
| `validation_outcome: validation_failed` | Table shape changed or profile stale — re-ingest |
| `charts_offered=1` but no UI chart | Frontend SSE / `ComputedChartsPanel` issue |

`chart_eligibility` in `RETRIEVAL_REQUEST_DETAIL` shows per-table outcomes.

### Table extraction issues

- Check `extra_metadata.table_qa` on indexed chunks in OpenSearch Dashboards
- Logs emit `TABLE_QA` lines with misalignment and fallback decisions
- Whitespace-aligned tables (e.g. 10-K filings) often trigger geometry/text-line recovery

### Image/chart retrieval

- Charts without nearby text or OCR are **skipped** at ingestion
- Install Tesseract for better keyword signal from chart labels
- Vector-drawn charts are detected separately from raster `page.images`

### Reindexing after config changes

Re-upload documents when you change:

- Embedding model/dimension or chunking strategy
- Chartability logic (`backend/app/charts/`) or unit detection
- **PDF text bbox / line_bboxes** logic (highlights require re-ingestion)

### Known limitations

- **No vision-model reasoning** — chart numeric Q&A depends on table/text chunks, not pixel understanding
- **Computed charts require chartable table retrieval** — asking for a "chart" does not bypass this
- **Many extracted "tables" are prose fragments** — only clean metric×period grids become chartable
- **English only** — embedding model and OCR tuned for English documents
- **DOCX Phase 1** — text + tables only; no embedded images; no in-app DOCX viewer or highlights
- **PDF highlights require re-ingestion** — documents indexed before bbox/`line_bboxes` support show the viewer but may lack overlays until re-uploaded
- **Highlight scope** — only sources from the **current answer** are highlighted, not the full document index
- **Answer LLM is single-turn** — chat history goes to the agent **router** only; the grounded answer step sees retrieved excerpts only
- **Agent router may use multiple Groq calls** per message (`AGENT_MAX_ROUNDS`) — latency scales with tool rounds
- **Legacy vs agent** — set `AGENT_ENABLED`; both paths share the same grounded answer prompt and sources UI
- **Async ingestion via BackgroundTasks** — not a durable job queue; large PDFs block one worker
- **In-process rate limits** — per-container; not shared across replicas without external store

---

## 19. PDF citation viewer and highlighting

The PDF citation viewer lets users jump from a retrieved source to the exact region in the original PDF, with **marker-pen style** highlight overlays. DOCX sources are out of scope for this viewer (see [§5](#5-ingestion-pipeline) DOCX section).

### End-to-end flow

```
Query answer completes
       │
       ▼
SSE `sources` event (bbox, line_bboxes, doc_id, page_count, …)
       │
       ▼
User clicks "Open page N in document" in SourcesPanel
       │
       ▼
PdfViewerPanel opens (fixed right side panel)
       │
       ├─► GET /config → pdf_viewer_page_window
       ├─► PDF.js loads GET /documents/{doc_id}/file (HTTP Range, Bearer token)
       ├─► Windowed canvas render (only pages near viewport ± window)
       └─► Per-line highlight rects scaled from PDF coords → viewport pixels
```

### Ingestion: where geometry comes from

**Location:** `backend/app/ingestion/text.py`

For each PDF **text** chunk, word-level coordinates from pdfplumber drive two stored fields:

| Field | Meaning |
|-------|---------|
| `bbox` | Union box covering all words in the chunk — used for scroll centering and fallback highlight |
| `extra_metadata.line_bboxes` | Array of `[x0, top, x1, bottom]` per visual line — **preferred** for rendering |

Lines are derived by clustering words with similar `top` values (`_group_lines()`). Per-line boxes avoid painting empty gutters between columns that a single union box would cover.

**Tables and images** store a table/image `bbox` but typically have no `line_bboxes`. The viewer falls back to the union `bbox` for those chunk types.

Coordinate system: **PDF points, top-origin** (`top`/`bottom` from pdfplumber). The frontend converts Y when mapping to PDF.js viewport space.

> **Re-ingestion required:** PDFs uploaded before this geometry was added will open in the viewer but may show **no highlights** until re-uploaded and re-indexed.

### Backend: PDF byte-range serving

**Location:** `backend/app/api/pdfs.py` → `GET /documents/{doc_id}/file`

| Behavior | Detail |
|----------|--------|
| Auth | Bearer token required; ownership check → **404** if not owned |
| Format | PDF only (resolved via `find_pdf_path()`) |
| Range | Parses `Range: bytes=start-end`; responds **206** with `Content-Range` |
| Streaming | 64 KiB read chunks; enables PDF.js lazy page fetch |

CORS exposes `Range` and `Content-Range` headers so the browser can stream from the Vite dev proxy.

### Frontend: PDF.js loading

**Location:** `frontend/src/pdf/pdfjs.ts`

- Worker: `pdfjs-dist/build/pdf.worker.min.mjs?worker` assigned to `GlobalWorkerOptions.workerPort` (fixes fake-worker failures in Vite)
- `disableAutoFetch: true` — opening page 340 does not download pages 1–339
- `rangeChunkSize: 65536` — matches backend chunk size
- Authenticated `httpHeaders: { Authorization: Bearer … }`

### Frontend: viewer panel

**Location:** `frontend/src/components/PdfViewerPanel.tsx`

| Feature | Implementation |
|---------|----------------|
| **Panel** | Fixed right drawer (`~52vw`, max 760px); zoom 50%–300%; fit-to-width baseline |
| **Virtualization** | Only renders canvas for pages within scroll viewport ± `PDF_VIEWER_PAGE_WINDOW` |
| **Doc lifecycle** | PDF document reloads only when `docId` changes — retargeting another citation in the same doc scrolls without tearing down PDF.js |
| **Canvas vs highlights** | Page rasterization effect depends on `pageNumber` + `scale` only; highlight geometry recomputes separately when `sources` or active citation changes |
| **Highlight membership** | Only `messageSources` from the **current answer** passed into the viewer — not all chunks in the document |
| **Primary citation** | Clicked source → amber fill (`bg-amber-300/25`) |
| **Other cited regions** | Same answer, other chunks on visible pages → sky fill (`bg-sky-300/15`) |
| **Style** | Fill-only marker pen — **no borders** |
| **Box selection** | `line_bboxes` when present; else single `bbox` rect |
| **Interaction** | Clicking a highlight rect sets it as primary and smooth-scrolls to center its page |

**Error isolation:** `PdfViewerBoundary.tsx` wraps the panel so a PDF.js render failure does not unmount the main chat UI.

### Configuration

| Setting | Default | Purpose |
|---------|---------|---------|
| `PDF_VIEWER_PAGE_WINDOW` | `2` | Extra pages rendered above/below the scroll viewport |

Exposed to the frontend via `GET /config` as `pdf_viewer_page_window`.

### Tests

`backend/tests/test_pdf_serving.py` — range parsing and 206 response behavior for owned PDFs.

---

## Quick reference: chunk lifecycle

```
PDF page                          DOCX body (w:p / w:tbl)
   │                                      │
   ├─► text words ──► overlap windows ──► content + bbox + line_bboxes
   ├─► table bbox   ──► table_to_markdown() ──► content (markdown table)
   └─► image/chart   ──► OCR + nearby text ──► content (proxy text)
                              │
DOCX paragraph ──► chunk_text() ──► content (prose, section metadata)
DOCX table     ──► table_to_markdown() ──► content (markdown table)
                              │
                              ▼
                        embed_texts(content)
                              │
                              ▼
                     OpenSearch rag_chunks
                     { content, embedding, bbox, extra_metadata }
                              │
                              ▼
              hybrid_search(BM25 + kNN) ← embed(query)
                   ▲                          │
                   │              ┌───────────┴───────────┐
         AGENT_ENABLED=false       │                       │
         (always retrieve)    text/table              image chunk
                   │               │                       │
         AGENT_ENABLED=true        │                       │
         router → tools ───────────┘                       │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        text/table      image chunk    chartable table
        → LLM context   → Sources UI   → chart spec → ComputedChartsPanel
              │               │                │
              └──► stream_groq_answer (grounded; excerpts only)
                        │
                        └──► sources SSE (bbox, line_bboxes, page_count, attached_images)
                                  │
                                  ▼
                        PDF viewer + hero strip + ComputedChartsPanel
```
