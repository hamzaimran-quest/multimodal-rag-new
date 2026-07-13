# Multimodal RAG — Workflow Guide

This guide explains how the system works end to end: **document ingestion** (PDF, DOCX, XLSX), **hybrid retrieval** in OpenSearch, **Groq tool-calling agent** for document Q&A, and the optional **SQL Agent** for live PostgreSQL queries — unified in one chat composer.

For original design decisions and open questions, see [ARCHITECTURE.md](./ARCHITECTURE.md). For the SQL Agent product plan, see [SQL_AGENT_PLAN.md](./SQL_AGENT_PLAN.md).

---

## Table of contents

1. [High-level overview](#1-high-level-overview)
2. [End-to-end data flow](#2-end-to-end-data-flow)
3. [Embeddings](#3-embeddings)
4. [OpenSearch](#4-opensearch)
5. [Ingestion pipeline](#5-ingestion-pipeline)
6. [Modality handling: text, tables, images](#6-modality-handling-text-tables-images)
7. [XLSX workbooks](#7-xlsx-workbooks)
8. [Retrieval (hybrid search)](#8-retrieval-hybrid-search)
9. [Query and answer generation](#9-query-and-answer-generation)
10. [SQL Agent](#10-sql-agent)
11. [Charts](#11-charts)
12. [Authentication, signup, and sessions](#12-authentication-signup-and-sessions)
13. [Per-user isolation](#13-per-user-isolation)
14. [Chat history](#14-chat-history)
15. [Security hardening](#15-security-hardening)
16. [API reference](#16-api-reference)
17. [Configuration](#17-configuration)
18. [Project layout](#18-project-layout)
19. [Running locally](#19-running-locally)
20. [Tuning and debugging](#20-tuning-and-debugging)
21. [Citation viewers (PDF, DOCX, XLSX)](#21-citation-viewers-pdf-docx-xlsx)
22. [Request logging](#22-request-logging)

---

## 1. High-level overview

This is a **multimodal document Q&A system** with an optional **SQL Agent** for user-provided PostgreSQL databases.

1. **Ingests** PDFs, DOCX, and XLSX — extracting text, tables, and (PDF/DOCX) images.
2. **Chunks** content into retrievable units (one table = one chunk; text split with overlap; XLSX row bands).
3. **Embeds** each chunk's `content` with a local sentence-transformer model (FastEmbed).
4. **Indexes** chunks in **OpenSearch** with BM25 + k-NN hybrid search.
5. **Routes** each chat message: schema-first **SQL** gate when a DB connection is active, then **RAG agent** for documents (or both for hybrid).
6. **Generates** grounded answers via **Groq** — streamed over SSE; document answers use **retrieved excerpts only**.
7. **Cites sources** (filename, location, chunk type, snippet) and opens format-specific viewers (PDF/DOCX/XLSX).
8. **Authenticates users** via email/password; each account has isolated documents, chats, SQL connections, and refresh sessions in **PostgreSQL**.

```
┌─────────────┐  signup/login   ┌──────────────┐
│  Frontend   │ ───────────────► │ PostgreSQL   │  users, chats, sql_connections, refresh tokens
│  (React)    │                 └──────────────┘
└──────┬──────┘
       │  upload (Bearer)         ┌──────────────────┐     index      ┌─────────────┐
       ├────────────────────────► │ Ingestion        │ ─────────────► │ OpenSearch  │
       │                          │ Pipeline         │  user-scoped   │ rag_chunks  │
       │                          └──────────────────┘                └──────┬──────┘
       │  query (SSE, Bearer)                                                    │
       ▼                                                                         │
┌─────────────┐     schema-first route + hybrid search (user_id filter)   ◄──────┘
│  Query API  │
└──────┬──────┘
       │
       ├──► SQL path (LangChain + Groq) ──► user PostgreSQL
       │
       └──► RAG agent (Groq tools) ──► OpenSearch retrieval ──► stream_groq_answer
                    │
                    └──► chat messages + sql_meta persisted to PostgreSQL
```

**Key design choices:**

| Area | Choice |
|------|--------|
| **Document formats** | PDF, DOCX, XLSX |
| **Answer model** | `GROQ_ANSWER_MODEL` (default `openai/gpt-oss-120b`) |
| **Aux / SQL / rewrite** | `GROQ_AUX_MODEL` (default `openai/gpt-oss-20b`) |
| **RAG routing** | Groq tool-calling agent (`search_documents`, `search_images`, `list_documents`, `create_chart`) |
| **SQL routing** | Schema cache + scope classifier **before** RAG agent; RAG agent runs with `sql_active=False` |
| **Image chunks** | Retrieved and shown in UI; **not** sent to answer LLM |
| **PDF highlights** | Per-line `line_bboxes` in PDF coordinate space |
| **DOCX viewer** | LibreOffice renders `__viewer.pdf`; bbox lookup for highlights |
| **XLSX viewer** | In-app spreadsheet panel with row/sheet highlights post-answer |
| **Isolation** | All data scoped by `user_id` from JWT; cross-tenant access → **404** |
| **Auth tokens** | Access JWT in memory; refresh in httpOnly cookie (`/auth` only) |

---

## 2. End-to-end data flow

### Upload → index

| Step | What happens | Code |
|------|--------------|------|
| 1 | User uploads **PDF, DOCX, or XLSX** via `POST /documents/upload` | `backend/app/api/documents.py` |
| 2 | Backend assigns `doc_id`, saves to `data/uploads/{user_id}/{doc_id}/` | `save_upload_file()` |
| 3 | Document record created in `rag_documents` with `user_id` (`status: processing`) | `create_document_record()` |
| 4 | Ingestion runs in FastAPI `BackgroundTasks` | `_schedule_ingestion()` |
| 5 | Format dispatch: PDF / DOCX / XLSX extractors | `pipeline.py` |
| 6 | All chunk `content` strings batch-embedded | `embed_texts()` |
| 7 | Old chunks for `doc_id` deleted; new chunks indexed with `user_id` | `index_chunks()` |
| 8 | Status → `indexed` with `chunk_count` and format-specific metadata | `update_document_record()` |

### Query → answer

**Entry:** `POST /query/stream` → `_agent_event_stream()` in `query.py` (agent mode is always used; there is no legacy always-on retrieval path).

| Step | What happens | Code |
|------|--------------|------|
| 1 | Authenticated user sends question (optional `session_id`, optional `doc_ids` scope) | `query.py` |
| 2 | User message appended to chat session | `chat/service.py` |
| 3 | If active SQL connection → `plan_sql_route()` (schema cache + scope classifier) | `sql_agent/routing.py` |
| 4 | SSE `route` event: `sql` \| `rag` \| `hybrid` | `query.py` |
| 5 | **SQL phase** (`sql`/`hybrid`): stream SQL agent tokens + `sql` provenance | `sql_agent/streaming.py` |
| 6 | **RAG phase** (`rag`/`hybrid`/`agent_only`): `iter_agent_turn()` with `sql_active=False` | `llm/agent.py` |
| 7 | Grounded answer stream (`event: token`); hybrid merges SQL text + RAG text | `stream_groq_answer()` |
| 8 | `sources`, `charts`, assistant message persisted (`sources`, `charts`, `sql_meta`) | `_persist_assistant_reply()` |

---

## 3. Embeddings

**Location:** `backend/app/ingestion/embeddings.py`

| Setting | Default | Notes |
|---------|---------|-------|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | FastEmbed (ONNX), no PyTorch |
| `EMBEDDING_DIMENSION` | `384` | Must match OpenSearch `knn_vector` mapping |

- Vectors are **L2-normalized** to unit length before indexing and query.
- **Ingestion** embeds every chunk `content`; **retrieval** embeds the user query.
- Changing model or dimension requires re-indexing all documents.

### Process

1. `get_embedding_model()` loads the FastEmbed `TextEmbedding` model on first use (`@lru_cache`).
2. `embed_texts(texts)` calls `model.embed(texts)` for a batch of strings.
3. Each vector is **explicitly L2-normalized** to unit length (matching sentence-transformers `normalize_embeddings=True`).
4. `assert_unit_vectors()` validates norms are ≈ 1.0 before returning.

```python
# Simplified flow
raw_vectors = model.embed(texts)
vectors = [l2_normalize(v) for v in raw_vectors]
```

**Critical:** Ingest and query must use the **same model and normalization**. Changing `EMBEDDING_MODEL` or `EMBEDDING_DIMENSION` requires recreating the OpenSearch index and re-uploading all documents.

---

## 4. OpenSearch

**Location:** `backend/app/opensearch/`

### Infrastructure

- **Image:** `opensearchproject/opensearch:2.19.0` (single-node, security disabled locally)
- **k-NN:** HNSW, L2 space
- **Bootstrap:** `wait_for_opensearch()` → `ensure_indices()` → `ensure_hybrid_search_pipeline()`
- **Legacy wipe:** indices missing `user_id` in mapping are deleted on startup

### Indices

#### `rag_chunks`

| Field | Type | Purpose |
|-------|------|---------|
| `chunk_id` | keyword | UUID |
| `doc_id`, `user_id`, `filename` | keyword | Identity + isolation |
| `page_number` | integer | PDF page, DOCX block ordinal, or XLSX sheet-related index |
| `chunk_type` | keyword | `text`, `table`, `image` |
| `content` | text | BM25 + embedding source |
| `embedding` | knn_vector (384) | Vector search |
| `bbox` | float[] | PDF/DOCX viewer geometry |
| `image_path` | keyword | Extracted image file |
| `extra_metadata` | object | `source_format`, `section`, `chart_profile`, `line_bboxes`, XLSX sheet/row fields, etc. |

#### `rag_documents`

Registry for ingestion status, `page_count`, `workbook_schema` (XLSX), etc.

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

### Example hybrid query shape

```python
query_vector = embed_texts([query])[0]

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

---

## 5. Ingestion pipeline

**Orchestrator:** `backend/app/ingestion/pipeline.py` → `run_ingestion()`

| Format | Extractor | Notes |
|--------|-----------|-------|
| `.pdf` | `extract_page_chunks()` | Per-page text, tables, images |
| `.docx` | `extract_docx_chunks()` + `extract_docx_image_chunks()` | Body-order; LibreOffice → `__viewer.pdf` + bbox lookup |
| `.xlsx` | `extract_xlsx_workbook()` | Adaptive row bands, workbook schema analysis |

**Shared path:** `embed_texts()` → `delete_chunks_for_document()` → `index_chunks()`.

### PDF per-page order

```
Page N
  ├── 1. Tables (+ bboxes, chart_profile)
  ├── 2. Text outside table bboxes → word-chunk with overlap
  └── 3. Images/charts (excluding table regions)
```

### Text chunking

**Location:** `backend/app/ingestion/chunking.py`

| Setting | Default |
|---------|---------|
| `CHUNK_MAX_WORDS` | 400 |
| `CHUNK_OVERLAP_WORDS` | 50 |

PDF text chunks store `bbox` and `extra_metadata.line_bboxes` for citation highlighting.

### DOCX

| Aspect | Detail |
|--------|--------|
| Library | `python-docx` |
| Method | `extraction_method: "docx_native"` |
| Scope | Body-order text, tables, inline images |
| Block order | Walks `w:p` / `w:tbl` in body XML (not separate paragraph lists) |
| `page_number` | 1-based block ordinal in reading order |
| `extra_metadata.section` | Latest `Title` / `Heading*` style |
| Tables | One table = one chunk; same markdown + `chart_profile` path as PDF |
| Viewer | `render_docx_to_pdf()` → `locate_chunks_in_viewer_pdf()` |
| Min match | `DOCX_VIEWER_MIN_MATCH_RATIO` (default 0.6) |

Citations show **section name** (e.g. `Lists`) or `part N`, not `p. N`.

### Progress stages

| Progress | Stage |
|----------|-------|
| 2% | Queued |
| 5–55% | Parsing PDF pages |
| 30% | Parsing DOCX / XLSX |
| 65% | Generating embeddings |
| 85% | Indexing chunks |
| 100% | Completed or failed |

---

## 6. Modality handling: text, tables, images

### Text (`chunk_type: "text"`)

| Aspect | Detail |
|--------|--------|
| **PDF extractor** | pdfplumber word-level extraction |
| **Method** | `extraction_method: "pdfplumber"` |
| **Content** | Plain prose from page, excluding table regions |
| **Chunking** | Paragraphs → overlapping word windows (400 words, 50 overlap) |
| **Geometry** | `bbox` union + `extra_metadata.line_bboxes` per visual line |
| **At query time** | Included in LLM context (trimmed by `LLM_CONTEXT_MAX_CHARS`) |

**PDF text flow** (`backend/app/ingestion/text.py`):

1. `page.extract_words()` filtered to words **outside** table/narrative exclusion bboxes (`_kept_words` — strict containment).
2. Words grouped into paragraphs via `structured_paragraphs_for_page()` (vertical gap heuristic).
3. Sliding windows: step = `max(CHUNK_MAX_WORDS - CHUNK_OVERLAP_WORDS, 1)`.
4. Each window gets `bbox` (union of words) and `line_bboxes` (one tight box per visual line via `_group_lines()` — new line when `|top - current_top| > height × 0.6`).
5. Paragraphs shorter than `MIN_TEXT_WORDS` are skipped.

**DOCX text:** body-order paragraphs via `python-docx`; `page_number` = 1-based block ordinal; `extra_metadata.section` from latest `Title` / `Heading*` style.

**XLSX text:** pipe-delimited slim rows in row bands (see [§7](#7-xlsx-workbooks)).

---

### Tables (`chunk_type: "table"`)

| Aspect | Detail |
|--------|--------|
| **Primary extractor** | pdfplumber `find_tables()` (PDF) |
| **Fallback extractors** | Geometry reconstruction → text-line reconstruction → Camelot (lattice/stream) |
| **Content format** | Markdown table (`\| col \| col \|`) |
| **Chunking** | **One table = one chunk** (never split across rows) |
| **At query time** | Included in LLM context; may drive computed charts |

#### PDF table extraction cascade

**Location:** `backend/app/ingestion/pdf_tables.py`, `table_geometry.py`

Per-page order in `extract_page_chunks()`:

1. Discover table region bboxes via `page.find_tables()` (deduped within 2 pt tolerance).
2. Build **text exclusion zones** = table bboxes + optional right-adjacent narrative bands (annual-report layouts where prose duplicates tabular facts to the right of a table).
3. For each candidate bbox, run `_extract_table_candidate()`:

```
Probe pdfplumber extract → QA metrics
     │
     ├── Geometry-first (always attempted on label-expanded bbox)
     │        ├── reconstruct_table_geometry()  — method "geometry"
     │        └── reconstruct_table_text_lines() — method "text_lines" (looser y-clustering)
     │
     ├── pdfplumber extract (if probe OK and NOT short-circuited)
     │
     └── Camelot lattice → stream (optional, with validation)
```

**Quality gating** before accepting pdfplumber output:

| Check | Threshold | Meaning |
|-------|-----------|---------|
| Column misalignment | `MISALIGNMENT_THRESHOLD` = **0.15** | Fraction of data rows with wrong column count |
| Semantic label loss | `data_row_count >= 4` AND `numeric_ratio >= 0.6` AND `label_empty_ratio >= 0.6` | Numeric table with empty label column |
| Label short-circuit | `label_empty_ratio >= 0.6` AND `numeric_data_ratio >= 0.6` | Scrambled text-layer labels |
| Merged header geometry | cell width ≥ median × **1.8** | Unusually wide cells suggesting merged headers |

If quality checks fail → geometry/text-line recovery cascade. Recovered tables must pass `validate_reconstructed_table()`:

- `label_nonempty_rate >= 0.9`
- `misalignment_ratio <= 0.15`

**Geometry reconstruction** (`table_geometry.py`):

- Expands bbox leftward by up to **15%** of page width for wrapped row labels.
- Clusters words into row bands (y_tolerance = median word height × **0.65**; text-line fallback uses × **0.9**).
- Infers label column vs data columns; builds markdown rows with metric labels + period columns.
- Detects first data row when row has metric label + ≥2 financial values.

**Metadata stored in `extra_metadata`:**

```json
{
  "extraction_method": "pdfplumber | geometry | text_lines | camelot | docx_native | markdown_table | slim_rows",
  "table_headers": ["Revenue", "2024", "2025"],
  "chart_profile": {
    "chartable": true,
    "orientation": "wide",
    "period_count": 2,
    "metric_count": 6,
    "suggested_chart_type": "bar",
    "period_labels": ["2024", "2025"],
    "value_axis_label": "CNY Million"
  },
  "table_qa": {
    "misalignment_ratio": 0.0,
    "semantic_label_loss": false,
    "fallback_triggered": false,
    "recovery_method": "geometry",
    "recovery_validated": true,
    "label_empty_ratio": 0.0,
    "numeric_data_ratio": 0.85
  }
}
```

`chart_profile` is set at ingestion when the table's shape reduces cleanly to metric-by-period series (see [§11](#11-charts)). Tables without a valid profile omit this field.

---

### Images / charts (`chunk_type: "image"`)

| Aspect | Detail |
|--------|--------|
| **Approach** | OCR-proxy (no vision LLM) |
| **PDF method** | `extraction_method: "ocr_proxy"` |
| **DOCX method** | `extraction_method: "docx_ocr_proxy"` |
| **Content** | Synthetic text: page/block context + nearby words + OCR output |
| **Storage** | PNG crops in `data/images/{user_id}/{doc_id}/` |
| **At query time** | **Not** sent to LLM; `image_url` in sources for UI |

**PDF detection** (`backend/app/ingestion/images.py`):

1. **Embedded raster images** — from `page.images`; skip if area < 8,000 px² or overlaps table bbox; crop at 150 DPI.
2. **Vector chart regions** — cluster pdfplumber `rects`, `lines`, `curves`; requires ≥ 20 vector objects, area ≥ 20,000 px², min 80×60 px.

**Building retrievable content:**

```
Page {N} image/chart context.
Nearby text: {words within 80px margin of bbox, up to 80 words}
OCR text: {pytesseract output, max 500 chars}
```

Skip decorative images when both nearby text and OCR are empty. OCR requires `pytesseract` + Tesseract; if unavailable, OCR returns `""` silently.

**DOCX images** (`docx_images.py`):

- Inline body images only (table-cell images skipped).
- Nearby text from ±1 body blocks; min **2,500** pixels.
- Saved as PNG under `data/images/{user_id}/{doc_id}/`.

**Image serving:** `GET /images/{doc_id}/{filename}` — ownership check → **404** if not owned. Frontend loads via `authFetch` + blob URLs (img tags cannot send Bearer headers).

---

### Modality comparison at a glance

| | Text | Table | Image |
|---|------|-------|-------|
| **Embedded field** | Prose | Markdown table | OCR + proximity text |
| **LLM context** | ✅ Yes | ✅ Yes | ❌ No (display only) |
| **BM25 searchable** | ✅ | ✅ | ✅ |
| **Vector searchable** | ✅ | ✅ | ✅ |
| **UI rendering** | Snippet + PDF/DOCX highlight | Snippet + optional chart | Thumbnail / hero strip |

---

## 7. XLSX workbooks

**Locations:** `backend/app/ingestion/xlsx_*.py`, `backend/app/retrieval/xlsx_expand.py`, `backend/app/ingestion/xlsx_highlight.py`

### Ingestion pipeline

**Entry:** `extract_xlsx_workbook()` in `xlsx_enrich.py` (orchestrates extract + schema + enrichment).

1. **Load workbook** (`xlsx_workbook.py`) — `data_only=True` (formulas → last computed values); visible sheets only.
2. **Per-sheet chunking** (`xlsx_extract.py`):
   - If sheet has Excel **Table objects** → one markdown chunk per table (`content_format: "markdown_table"`).
   - Else → **row bands** over used range; row 1 = header; pipe-delimited slim rows (`content_format: "slim_rows"`).
3. **Adaptive band sizes** (`xlsx_serialize.resolve_row_band_size`):

| Column count | Env var | Default band size |
|--------------|---------|-------------------|
| ≥ 10 (wide) | `EXCEL_ROW_BAND_SIZE_WIDE` | 10 |
| ≥ 6 (medium) | `EXCEL_ROW_BAND_SIZE_MEDIUM` | 15 |
| else (narrow) | `EXCEL_ROW_BAND_SIZE` | 30 |

4. **Workbook schema** (`xlsx_schema.py`, gated by `EXCEL_SCHEMA_ENABLED`):
   - Aux LLM proposes sheet joins, key columns, and `one_to_many` / `soft_link` relationships.
   - Deterministic validation: overlap ≥ **0.9** for hard links; **0.85–0.9** for soft links; rejects missing keys/columns.
   - Stored on `rag_documents.workbook_schema`.
5. **Enrichment** (`xlsx_enrich.py` — Approach C):
   - **Primary sheets:** satellite payload columns appended; `one_to_many` → `_summary` columns (preview 3 values + "+N more").
   - **Satellite sheets:** separate banded chunks with `sheet_role: "satellite"`.
   - **Standalone sheets:** native extraction + FK annotation with `sheet_role: "standalone"`.

**Chunk metadata examples:**

```json
{
  "source_format": "xlsx",
  "sheet_name": "Orders",
  "sheet_index": 2,
  "sheet_role": "primary",
  "row_range": [2, 31],
  "col_range": [1, 8],
  "entity_keys": ["customer_id"],
  "row_entity_keys": {"5": {"customer_id": "42"}},
  "table_headers": ["Order ID", "Customer", "Amount"]
}
```

### Retrieval

- Scoped or XLSX-heavy queries cap at `EXCEL_TOP_K` (default **5**).
- **Entity-key expansion** (`xlsx_expand.py`):
  - Resolves anchor keys from top hit(s); cap `EXCEL_ANCHOR_MAX_ENTITIES` (default **3**).
  - Fetches linked sheet chunks per anchor (clusters + soft links + standalone FK links).
  - Expanded chunks tagged `xlsx_anchor_expanded=True`, `score=0.0`.
  - Legacy bulk expand skipped when multi-sheet workbook + query has ≥2 tokens.
- **`limit_xlsx_chunks()`** — deduplicates row bands; prioritizes satellite/standalone over duplicate primary bands.

### Post-answer highlighting

- `apply_xlsx_highlights_to_sources()` — maps answer entities to `highlight_row` on matching sources.
- Frontend **SpreadsheetViewerPanel** — sheet tabs, grid with row numbers; highlights only on `sheet_role: "primary"` when active sheet matches.
- Row ranges ≤ **24** rows get full highlight band; wider ranges scroll only.

---

## 8. Retrieval (hybrid search)

**Service:** `backend/app/retrieval/service.py` → `hybrid_retrieve()`  
**Search:** `backend/app/opensearch/search.py` → `hybrid_search()`

### Algorithm

BM25 `match` on `content` + k-NN on `embedding`, fused via `hybrid-search-pipeline`. Always filtered by `user_id`; optional `doc_id` / `doc_ids` scope.

### Parameters

| Param | Default | Notes |
|-------|---------|-------|
| `DEFAULT_TOP_K` | 5 | General hybrid search |
| `PDF_TOP_K` | 7 | PDF-only scoped searches |
| `PDF_TABLE_SLOTS` | 3 | Reserved table slots for PDF scope |
| `EXCEL_TOP_K` | 5 | XLSX retrieval cap |
| `LLM_CONTEXT_MAX_CHARS` | 12000 | Context trim before answer LLM |

### Scoping

- Frontend sends `doc_ids` (or legacy single `doc_id`) on `/query/stream`.
- `validate_scope_doc_ids()` verifies ownership; invalid IDs → **404**.
- `scope_filenames()` injected into agent router prompt.

### Table slots (PDF)

**Location:** `backend/app/retrieval/table_slot.py`

When scope is **PDF-only** (`pdf_scope=True`):

1. Run primary hybrid search (text + all types).
2. Run separate hybrid search with `chunk_type="table"` (pool size `PDF_TABLE_CANDIDATE_POOL`, default **8**).
3. Reserve up to `PDF_TABLE_SLOTS` (default **3**) table chunks in merged results.
4. Fill remaining slots with top text hits (`top_k - len(chosen_tables)`).

No keyword gating — the LLM disambiguates among multiple tables in context.

### Context trimming

**Location:** `backend/app/retrieval/context.py`

- `select_chunks_for_llm_context()` — keeps text/table chunks under `LLM_CONTEXT_MAX_CHARS` (default **12000**).
- Priority sort: `sheet_role` rank (standalone < satellite < primary), then score desc, then content length asc.
- Image chunks preserved in result list but **not** counted toward char budget.
- XLSX chunks serialized via `format_chunk_content_for_llm()`; DOCX shows `Section:` instead of `Page:` in context blocks.

### Image attachment

| Track | When | Mechanism |
|-------|------|-----------|
| **Intent (A)** | User explicitly asks to see a photo/chart | `search_images` agent tool |
| **PDF proximity (B)** | Implicit relevance (e.g. "who is the chairman") | `resolve_proximity_attachments()` — bbox/column overlap scoring |
| **DOCX proximity (C)** | Text/table anchor near inline image | `resolve_docx_proximity_attachments()` — ±`DOCX_IMAGE_PROXIMITY_BLOCK_RADIUS` blocks |

Post-retrieval enrichment in `_assemble_retrieval_payload()`:

- `build_display_images()` — dedup by IoU (`IMAGE_DEDUP_IOU`, default 0.6), cap `IMAGE_MAX_DISPLAY` (default 2).
- Hero cap: 1 image when `search_images` / visual intent required; else 2.
- **`visual_note`** — short hint to answer LLM when an image is shown separately (prevents "not found" when hero image is visible).

Image-only agent results still resolve `page_count` on sources from the document registry so the PDF viewer can open correctly.

### Result mapping

Each hit becomes a `RetrievedChunk` with `chunk_id`, `doc_id`, `filename`, `page_number`, `chunk_type`, `content`, `score`, `bbox`, `image_url`, `extraction_method`, and full `extra_metadata`.

### Request logging

See [§22](#22-request-logging).

### Query rewrite

Before routing, `rewrite_retrieval_query()` (aux Groq, `QUERY_REWRITE_ENABLED`) rewrites follow-ups using prior user queries + last assistant reply (`CHAT_HISTORY_TURNS`, `CHAT_LAST_REPLY_MAX_CHARS`).

### Standalone search

`GET/POST /search` — hybrid retrieval without LLM generation.

---

## 9. Query and answer generation

**Locations:** `backend/app/api/query.py`, `backend/app/llm/agent.py`, `backend/app/llm/groq.py`

### Unified flow

```
User message
     │
     ▼
Query rewrite (optional)
     │
     ▼
Active SQL connection? ──no──► RAG agent only (route: rag)
     │
    yes
     ▼
plan_sql_route() — schema cache + scope classifier
     │
     ├── sql ──────► SQL agent stream ──► (done or continue if hybrid)
     ├── rag ──────► RAG agent only
     ├── hybrid ───► SQL agent stream ──► RAG agent + sql_context_note
     └── agent_only ► small talk via RAG agent (no SQL)
     │
     ▼
iter_agent_turn() — tools WITHOUT query_database (sql_active=False)
     │
     ▼
stream_groq_answer() — grounded on retrieved excerpts (+ sql_context_note if hybrid)
```

### RAG agent tools

| Tool | Implementation | Returns |
|------|----------------|---------|
| `search_documents` | `hybrid_retrieve()` (+ table slots for PDF scope) | Text/table/image-metadata chunks |
| `search_images` | `retrieve_intent_images()` | Top image chunk(s) for explicit visual queries |
| `list_documents` | `list_document_records()` | User's files, status, chunk counts |
| `create_chart` | Aux LLM → Chart.js → QuickChart.io URL | Chart image URL + citation |

`query_database` is **not** registered on the RAG router during `/query/stream` (`sql_active=False`) — SQL is handled by the dedicated SQL phase above.

### Multi-round routing

The router may call tools across up to `AGENT_MAX_ROUNDS` Groq rounds (default **1**):

- Round 1: `list_documents` when scope is unclear, or `search_documents` with refined query.
- Higher `AGENT_MAX_ROUNDS` allows refine-and-re-search loops.

Chunks from all rounds are **merged** (`_merge_retrieved_chunks`, deduped by `chunk_id`, best score wins) before the answer step.

### Follow-ups and query rewriting

- **`rewrite_retrieval_query()`** runs before the router when `QUERY_REWRITE_ENABLED=true` — uses prior user queries (`CHAT_HISTORY_TURNS`, default 6) + last assistant reply (`CHAT_LAST_REPLY_MAX_CHARS`, default 800).
- Router prompt instructs rewriting follow-ups into standalone queries.
- **Forced-search fallback** — if router answers a document question in plain text without tools, backend automatically runs `search_documents` with `anchor_fallback_query` for pronoun follow-ups.

### Route outcomes

| Outcome | Behavior |
|---------|----------|
| Greeting / small talk | `direct_answer` from router (no search) |
| Ambiguous scope | `is_clarification` question (no search) |
| Document Q&A | `search_documents` → grounded `stream_groq_answer()` |
| Visual request | `search_images` (+ often `search_documents`) |
| Chart request | `create_chart` only (no `search_documents`) |

Chat history is sent to the **router only**. The **answer** LLM never sees prior turns — only retrieved excerpts — to prevent hallucination from world knowledge or stale chat context.

### Hallucination guards

| Guard | Behavior |
|-------|----------|
| Router prompt | Must not invent document facts; must call tools for factual Q&A |
| Forced search | Non-greeting document questions without tool calls → `search_documents` |
| Answer prompt (`SYSTEM_PROMPT` in `groq.py`) | Never use general world knowledge; only excerpts; missing fact → `"Not found in the provided documents"` |
| Grounding path | Agent answers always use `stream_groq_answer()`, not router direct text (except greetings/clarification) |
| Inline table guard | Logs `LLM_ANSWER_INLINE_TABLE` when model embeds `\| --- \|` inside bullets |

### Groq tool-call recovery

If Groq returns malformed `tool_use_failed` output with `<function=name{...}>` syntax, `_recover_tool_calls_from_failed_generation()` parses it into valid tool calls.

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

| Setting | Default | Role |
|---------|---------|------|
| `GROQ_ANSWER_MODEL` | `openai/gpt-oss-120b` | Grounded answer streaming |
| `AGENT_MODEL` | `openai/gpt-oss-120b` | Router tool-calling |
| `GROQ_AUX_MODEL` | `openai/gpt-oss-20b` | Query rewrite, charts, SQL scope classifier |
| Temperature | 0.1 | Answer + router |

### SSE persistence note

The agent loop runs **inside** the SSE generator. ORM objects from the request scope are detached, so:

- `user_id` / `chat_id` captured as integers before streaming.
- SQL connection fields snapshotted via `snapshot_connection()` before DB session closes.
- Assistant messages saved via `_persist_assistant_reply()` using a **fresh** `SessionLocal()` session.

### SSE event stream

| Event | Payload | When |
|-------|---------|------|
| `meta` | `{ query, top_k, doc_id, doc_ids, session_id, agent: true }` | Start |
| `route` | `{ mode: "sql" \| "rag" \| "hybrid" }` | After SQL route planning |
| `tool` | `{ name, status, round?, label? }` | SQL internal tools + RAG agent tools |
| `token` | `{ token }` | Streaming answer (SQL + RAG) |
| `sql` | `{ connection_id, display_name, queries[] }` | SQL phase complete |
| `sources` | `{ sources: [...] }` | After answer (with XLSX highlights) |
| `charts` | `{ charts: [...] }` | When charts produced |
| `error` | `{ message }` | On failure |
| `done` | `{ ok }` | End |

**Source fields include:** `chunk_id`, `doc_id`, `filename`, `page_number`, `viewer_page`, `chunk_type`, `snippet`, `image_url`, `score`, `source_format` (`pdf`/`docx`/`xlsx`), `section`, `bbox`, `line_bboxes`, `page_count`, `attached_images`, `sheet_name`, `row_range`, `highlight_row`, etc.

### Frontend (chat)

| Feature | Behavior |
|---------|----------|
| **Views** | Chat, Documents, SQL Agent |
| **Scope** | Sidebar checkboxes: specific docs or all |
| **Route badge** | "From database" / "From documents" / "Database + documents" |
| **Agent status** | Placeholder during tools ("Searching documents…", "Running database query…", etc.) |
| **Panels** | Sources, Charts, SQL provenance (collapsible) |
| **Viewers** | PDF/DOCX side panel, XLSX spreadsheet panel |

---

## 10. SQL Agent

**Locations:** `backend/app/sql_agent/`, `backend/app/api/sql_agent.py`

Optional per-user **PostgreSQL** connections. When an **active** connection exists, `/query/stream` runs schema-first routing before the RAG agent.

### Connections (`/sql-agent/*`)

| Feature | Detail |
|---------|--------|
| Storage | `user_sql_connections` in app PostgreSQL |
| Dialect | PostgreSQL only |
| Credentials | Fernet-encrypted (`SQL_CREDENTIALS_KEY` or derived from `JWT_SECRET`) |
| Active | Exactly one `is_active=true` per user |
| Schema cache | `schema_cache`, `schema_cache_fingerprint`, `schema_cached_at` on connection row |

**API:** list, add, activate, test, patch metadata/credentials, delete, deactivate all. All endpoints scoped by `user_id` → cross-user **404**.

### Schema cache (`schema_cache.py`, `schema_fetch.py`)

Three-tier cache:

```
1. In-memory (TTL SQL_CONNECTION_CACHE_TTL_SECONDS, default 300s)
2. PostgreSQL JSON on UserSqlConnection (schema_cache, schema_cache_fingerprint, schema_cached_at)
3. Live introspection via SQLAlchemy inspector
```

- **Digest format:** `table(col:type, ...)` per table; capped at `SQL_SCHEMA_MAX_TABLES` (40) and `SQL_SCHEMA_MAX_CHARS` (12000).
- **Fingerprint:** SHA-256 of connection URL — invalidates cache on credential change.
- **Warm:** on add, activate, credential update. **Invalidate:** on credential replace or delete.
- Injected into SQL agent prompt; listing tools disabled when digest present.

### Scope classifier (`scope_classifier.py`)

**LLM path** (`SQL_SCOPE_CLASSIFIER_ENABLED=true`): aux Groq JSON `{ decision, confidence, matched_tables, reason }`.

**Heuristic fallback:**

| Signal | Route | Confidence |
|--------|-------|------------|
| Document keywords (pdf, upload, document, …) | `rag` | 0.8 |
| Aggregate SQL patterns (count, sum, how many, …) | `sql` | 0.7 |
| Entity keyword matching table name | `sql` | 0.78 |
| Table name token overlap | `sql` | 0.72 |

**Confidence gate:** `sql` with confidence < `SQL_SCOPE_MIN_CONFIDENCE` (0.55) → `rag` unless entity signal. Small talk → `agent_only`.

### SQL execution (`agent.py`, `streaming.py`, `prompts.py`)

- LangChain `create_sql_agent` / tool-calling agent with `ChatGroq` (`SQL_AGENT_MODEL` or aux model fallback).
- Schema cached → only `sql_db_query` + `sql_db_query_checker`; `sample_rows_in_table_info=0`.
- Prompt: read-only SELECT; no SQL in answer; no placeholder data; concise answers.
- Limits: `SQL_AGENT_MAX_STEPS` (10), `SQL_AGENT_QUERY_TIMEOUT_SECONDS` (30), `SQL_AGENT_MAX_ROWS` (100).

**Streaming:**

1. Tool placeholders via SSE (`label`: "Checking SQL query", "Running database query").
2. Tool-planning LLM output not streamed.
3. Final answer streams live after `sql_db_query` completes.
4. `clean_sql_answer_text()` strips echoed SQL.
5. Queries from intermediate steps — `sql_db_query` only, deduplicated.

**Hybrid:** `sql_context_note` passed to RAG answer step after SQL phase.

### Connection lifecycle

| Action | Behavior |
|--------|----------|
| Add | Test `SELECT 1` → encrypt → save; first connection auto-activates; warm schema |
| Activate | Deactivates others; warm schema |
| Update credentials | Re-test → invalidate → warm |
| Delete / deactivate | Clear active state; invalidate cache |

### Demo database (Docker)

`dvdrental-postgres` service on host port **5434**. Connection URL from backend container:

```
postgresql://dvdrental:dvdrental@dvdrental-postgres:5432/dvdrental
```

---

## 11. Charts

Charts are **not** triggered by query wording alone for structural charts — retrieval of a chartable table chunk is required. The `create_chart` tool and auto-chart path additionally require chart-intent keywords.

### A. Structural computed charts (ingestion `chart_profile`)

**Locations:** `backend/app/charts/profile.py`, `spec.py`, `service.py`, `units.py`

#### When a chart appears (structural path)

```
Ingestion:  table shape → chart_profile (chartable: true/false)
Query:      hybrid search retrieves chunk → re-validate → emit chart spec
```

Both must be true:

1. Table marked **`chart_profile.chartable`** at ingestion.
2. That table chunk appears in retrieval for the query.

#### Chart types

| Type | Structural rule | UI |
|------|-----------------|-----|
| **Bar / grouped bar** | 1–8 metrics, 2–5 periods | Colored grouped bars (SVG) or QuickChart |
| **Line** | 1 metric, 3–5 periods | Single line + points |
| **(none)** | Validation fails | LLM may still answer using table markdown |

#### Ingestion-time analysis (`analyze_table_chartability`)

| Threshold | Value |
|-----------|-------|
| `MIN_PERIODS_BAR` | 2 |
| `MIN_PERIODS_LINE` | 3 |
| `MIN_METRICS` | 1 |
| Composition row rejection | values sum 90–110, all 0–100, ≥3 values (pie-like) |
| Orientation ambiguity | wide vs long scores within **0.25** → not chartable |

- **Wide layout** — metrics in rows, periods in column headers.
- **Long layout** — periods in row labels, metrics in columns.
- Annotation columns excluded (short acronym headers ≤3 chars, or columns dominated by `%` values).
- **`detect_value_axis_label()`** reads parenthetical units from headers: `(CNY Million)`, `%`, etc.; fails closed to `"Value"` when contradictory.

#### Query-time validation (`validate_and_build_chart_spec`)

1. Re-parses table markdown from chunk `content`.
2. Re-runs chartability; must match stored profile counts/orientation.
3. Numbers come **only** from parsed cells — never from the LLM.

### B. LLM + QuickChart (`create_chart` tool / auto-chart)

**Locations:** `backend/app/charts/build.py`, `quickchart.py`, `auto.py`, `llm/tools.py`

- Agent calls `create_chart` when user asks for chart/graph/plot/visualize.
- **`try_auto_chart_from_retrieval()`** — if chart keywords detected and table chunk retrieved, auto-invokes chart builder.
- **`attempt_chart_from_chunk()`**:
  - XLSX → `build_excel_chart_data_spec_from_chunk` (`spec_source: "excel_entity_grid"`).
  - PDF → structural profiling first, LLM fallback (`extract_chart_data_spec`).
- Aux LLM builds Chart.js config → **QuickChart.io** image URL (`chart_url` in SSE).
- `chart_note` injected into answer prompt so model doesn't emit plotting code.
- `derivation: "tool"` on chart objects.

| Setting | Default |
|---------|---------|
| `CHART_LLM_MODEL` | `openai/gpt-oss-20b` |
| `CHART_TABLE_MAX_CHARS` | 4000 |
| `QUICKCHART_WIDTH` / `HEIGHT` | 500 / 300 |

### PDF image vs computed chart

| | PDF image chunk | Computed / QuickChart |
|---|-----------------|----------------------|
| **Source** | Raster/vector crop | Parsed table numbers |
| **Trigger** | Image chunk in top-k | Chartable table or `create_chart` tool |
| **LLM input** | No | No |
| **UI** | Sources panel thumbnail | `ComputedChartsPanel` below answer |

---

## 12. Authentication, signup, and sessions

**Locations:** `backend/app/api/auth.py`, `backend/app/auth/`, `frontend/src/pages/`, `frontend/src/auth/`

Users sign up with **email + password** (bcrypt, cost factor 12).

| Token | Transport | TTL | Purpose |
|-------|-----------|-----|---------|
| **Access JWT** | `Authorization: Bearer` | 15 min | API auth; `sub` = `user_id` |
| **Refresh** | httpOnly cookie `rag_refresh` (path `/auth` only) | 7 days | Silent renewal; SHA-256 hash in PostgreSQL |

Refresh cookie is **not** sent to `/documents`, `/query`, `/images`, or `/sql-agent`.

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/auth/signup`, `/auth/login` | Create session |
| `POST` | `/auth/refresh` | Rotate refresh token (old token revoked) |
| `POST` | `/auth/logout` | Revoke refresh |
| `GET` | `/auth/me` | Current user |
| `POST` | `/auth/change-password` | Change password; revoke **all** sessions |

**Frontend:** access token in memory (`tokenStore.ts`); `authFetch` auto-refreshes on 401; `RequireAuth` guards `/`.

**UI:** Login (typewriter headline + form), Signup (form + feature carousel); dark gradient layout.

---

## 13. Per-user isolation

| Layer | Mechanism |
|-------|-----------|
| OpenSearch | `user_id` filter on every query |
| Filesystem | `data/uploads/{user_id}/`, `data/images/{user_id}/` |
| Chats | `chat_sessions.user_id` |
| SQL connections | `user_sql_connections.user_id` |
| Images API | Ownership check → **404** |

Cross-user `doc_id` access returns **404** (not 403).

### Ownership checks

| Action | Behavior when `doc_id` belongs to another user |
|--------|-----------------------------------------------|
| `GET /documents/{doc_id}/status` | **404** |
| `DELETE /documents/{doc_id}` | **404** |
| `GET/POST /search?doc_id=...` | **404** |
| `POST /query/stream` with scoped `doc_ids` | **404** |
| `GET /images/{doc_id}/{filename}` | **404** |
| `GET /sql-agent/connections/{id}` | **404** |

### Migration / legacy data

On startup, `ensure_indices()` deletes OpenSearch indices lacking `user_id` mapping and may wipe legacy upload/image dirs. Users re-upload after deploy.

---

## 14. Chat history

**Locations:** `backend/app/api/chats.py`, `backend/app/chat/`

### Tables

| Table | Key fields |
|-------|------------|
| `chat_sessions` | `id`, `user_id`, `title`, timestamps |
| `chat_messages` | `role`, `content`, `sources` (JSON), `charts` (JSON), `sql_meta` (JSON) |

### Flow

1. `POST /query/stream` accepts optional `session_id`; creates session if omitted.
2. User message saved before stream.
3. Assistant message saved after `done: ok` with `content`, `sources`, `charts`, `sql_meta`:

```json
{
  "sql_meta": {
    "connection_id": 1,
    "display_name": "DVD Rental",
    "queries": ["SELECT ... FROM film WHERE ..."],
    "route_mode": "sql"
  }
}
```

Reloaded chats restore sources, charts, and SQL provenance panel from `sql_meta`.

### History at query time

- **Query rewrite:** prior user queries + last assistant reply.
- **Agent router:** rewritten query + scope hint (not full answer history in grounding step).
- **Chart follow-ups:** `prior_table_chunk_ids` from recent assistant sources.

---

## 15. Security hardening

| Control | Implementation |
|---------|----------------|
| Rate limiting | Upload 10/min, query 30/min per `user_id` → **429** |
| Refresh rotation | Old token revoked on each refresh |
| Password change | Revokes all refresh tokens |
| CORS | `CORS_ORIGINS` env |
| JWT secret | `REQUIRE_SECURE_JWT_SECRET` blocks insecure defaults in production |
| SQL credentials | Encrypted at rest; never returned in API responses |

---

## 16. API reference

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `GET` | `/health` | — | Health check |
| `GET` | `/config` | — | Public config (e.g. `pdf_viewer_page_window`) |
| `POST` | `/auth/signup`, `/login`, `/refresh`, `/logout` | varies | Auth |
| `GET` | `/auth/me` | Bearer | Current user |
| `POST` | `/auth/change-password` | Bearer | Password change |
| `POST` | `/documents/upload` | Bearer | Upload PDF/DOCX/XLSX |
| `GET` | `/documents` | Bearer | List user's documents |
| `GET` | `/documents/{doc_id}/status` | Bearer | Ingestion progress |
| `GET` | `/documents/{doc_id}/file` | Bearer | PDF byte-range serving |
| `DELETE` | `/documents/{doc_id}` | Bearer | Delete document + chunks |
| `GET/POST` | `/search` | Bearer | Hybrid retrieval only |
| `POST` | `/query/stream` | Bearer | Full pipeline (SQL + RAG + SSE) |
| `GET/POST` | `/chats` | Bearer | List / create sessions |
| `GET/DELETE` | `/chats/{id}` | Bearer | Load / delete session |
| `GET` | `/images/{doc_id}/{filename}` | Bearer | Extracted images |
| `GET` | `/sql-agent/status` | Bearer | SQL connection status |
| `GET` | `/sql-agent/connections` | Bearer | List connections |
| `POST` | `/sql-agent/connections` | Bearer | Add connection |
| `POST` | `/sql-agent/connections/{id}/activate` | Bearer | Set active |
| `POST` | `/sql-agent/connections/{id}/test` | Bearer | Test connection |
| `PATCH` | `/sql-agent/connections/{id}` | Bearer | Update name/description |
| `PATCH` | `/sql-agent/connections/{id}/credentials` | Bearer | Replace URL |
| `DELETE` | `/sql-agent/connections/{id}` | Bearer | Delete connection |
| `POST` | `/sql-agent/deactivate` | Bearer | Deactivate all |

Interactive docs: `http://localhost:8000/docs`

---

## 17. Configuration

All settings in `backend/app/config.py`, loaded from `.env`. See `.env.example` for the full template.

### Core

```bash
OPENSEARCH_HOST=opensearch
OPENSEARCH_PORT=9200
GROQ_API_KEY=your_groq_api_key_here
GROQ_ANSWER_MODEL=openai/gpt-oss-120b
GROQ_AUX_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
DEFAULT_TOP_K=5
LLM_CONTEXT_MAX_CHARS=12000
```

### Agent & rewrite

```bash
AGENT_MODEL=openai/gpt-oss-120b
AGENT_MAX_ROUNDS=1
QUERY_REWRITE_ENABLED=true
CHAT_HISTORY_TURNS=6
CHAT_LAST_REPLY_MAX_CHARS=800
```

### SQL Agent

```bash
SQL_AGENT_ENABLED=true
SQL_AGENT_MODEL=openai/gpt-oss-20b
SQL_AGENT_MAX_STEPS=10
SQL_SCHEMA_MAX_TABLES=40
SQL_SCOPE_CLASSIFIER_ENABLED=true
SQL_SCOPE_MIN_CONFIDENCE=0.55
SQL_CREDENTIALS_KEY=
```

### XLSX, images, charts, auth

See `.env.example` for `EXCEL_*`, `IMAGE_*`, `DOCX_*`, `CHART_*`, `QUICKCHART_*`, `DATABASE_URL`, `JWT_SECRET`, `CORS_ORIGINS`, rate limits.

**Never commit `.env`** — it is gitignored.

---

## 18. Project layout

```
multimodal-rag/
├── docker-compose.yml          # OpenSearch, PostgreSQL, dvdrental, backend, frontend
├── .env.example
├── ARCHITECTURE.md
├── RAG_GUIDE.md                # This document
├── SQL_AGENT_PLAN.md
├── data/
│   ├── images/{user_id}/{doc_id}/   # Extracted chart/image PNGs
│   └── uploads/{user_id}/{doc_id}/  # Uploaded PDFs, DOCX, XLSX (+ __viewer.pdf for DOCX)
├── backend/
│   └── app/
│       ├── main.py             # FastAPI app, CORS, lifespan
│       ├── config.py           # Settings
│       ├── db/
│       │   ├── models.py       # User, RefreshToken, ChatSession, ChatMessage, UserSqlConnection
│       │   └── session.py      # SQLAlchemy session + init_db() migrations
│       ├── auth/
│       │   ├── service.py      # Users, refresh tokens, password change
│       │   ├── security.py     # bcrypt, JWT, refresh token hashing
│       │   ├── dependencies.py # get_current_user
│       │   ├── rate_limit.py   # Per-user upload/query rate limits
│       │   └── schemas.py
│       ├── chat/
│       │   ├── service.py      # Session CRUD, message persistence
│       │   └── schemas.py
│       ├── api/
│       │   ├── auth.py, chats.py, documents.py, pdfs.py, images.py
│       │   ├── query.py        # SSE unified pipeline (SQL + RAG)
│       │   ├── search.py, spreadsheet.py, sql_agent.py
│       │   └── health.py
│       ├── ingestion/
│       │   ├── pipeline.py     # Format dispatch
│       │   ├── text.py         # PDF text + line_bboxes
│       │   ├── pdf_tables.py, table_geometry.py, tables.py
│       │   ├── docx_extract.py, docx_render.py, docx_bbox_lookup.py, docx_images.py
│       │   ├── xlsx_extract.py, xlsx_workbook.py, xlsx_schema.py, xlsx_enrich.py, xlsx_highlight.py
│       │   ├── images.py, chunking.py, embeddings.py
│       │   └── models.py
│       ├── opensearch/
│       │   ├── client.py, bootstrap.py, indices.py, pipelines.py
│       │   ├── chunks.py, documents.py, search.py
│       ├── charts/
│       │   ├── profile.py, spec.py, build.py, quickchart.py, auto.py, service.py
│       ├── retrieval/
│       │   ├── service.py, scope.py, table_slot.py, xlsx_expand.py
│       │   ├── context.py, image_attach.py, docx_image_attach.py
│       │   └── request_log.py
│       ├── llm/
│       │   ├── groq.py, agent.py, tools.py, rewrite.py
│       └── sql_agent/
│           ├── agent.py, streaming.py, routing.py, scope_classifier.py
│           ├── schema_cache.py, schema_fetch.py, service.py, crypto.py, prompts.py
└── frontend/src/
    ├── main.tsx                # Router: /login, /signup, / (guarded)
    ├── App.tsx                 # chat | docs | sql views
    ├── auth/                   # AuthContext, tokenStore, RequireAuth
    ├── api/                    # auth, chats, client, query, sqlAgent, http
    ├── pdf/                    # pdfjs.ts, log.ts
    ├── lib/                    # heroImages.ts, spreadsheet.ts
    └── components/
        ├── ChatAssistantMessage.tsx, SqlAgentPanel.tsx, SqlProvenancePanel.tsx
        ├── PdfViewerPanel.tsx, PdfViewerBoundary.tsx
        ├── SpreadsheetViewerPanel.tsx
        ├── SourcesPanel.tsx, ComputedChartsPanel.tsx, HeroImages.tsx
        └── MarkdownAnswer.tsx, IngestionProgressRing.tsx
```

---

## 19. Running locally

### Docker Compose (recommended)

```bash
cp .env.example .env
# Set GROQ_API_KEY

docker compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| OpenSearch | http://localhost:9200 |
| OpenSearch Dashboards | http://localhost:5601 |
| App PostgreSQL | localhost:5432 (`rag` / `rag` / `rag`) |
| DVD Rental demo DB | localhost:5434 (`dvdrental` / `dvdrental` / `dvdrental`) |

### Verify end-to-end

1. Sign up at http://localhost:5173/signup
2. Upload PDF, DOCX, or XLSX → wait for `indexed`
3. (Optional) SQL Agent tab → add DVD Rental connection URL above → activate
4. Ask a document question → route badge + sources
5. Ask a database question (e.g. film lookup) → `route: sql`, SQL provenance panel
6. Ask a hybrid question → `route: hybrid`, SQL text then document answer
7. Confirm chat persists in sidebar Conversations

### Test retrieval without LLM

```bash
curl -H "Authorization: Bearer <token>" \
  "http://localhost:8000/search?query=revenue+2025"
```

### Backend tests

```bash
# Inside backend container
docker exec multimodal-rag-backend pytest -v

# Or test profile (subset)
docker compose --profile test run backend-tests
```

**Coverage areas** (~76 test files): chunking, table geometry, PDF/DOCX/XLSX ingestion, auth isolation, SQL agent routing/schema/crypto, agent tools, charts, image attach, query API.

**Frontend tests:** `npm test` in `frontend/` (Vitest + Testing Library).

---

## 20. Tuning and debugging

### Retrieval quality

| Knob | Where to change |
|------|-----------------|
| BM25 vs vector balance | `weights` in `pipelines.py` (default 0.5/0.5) |
| Result count | `DEFAULT_TOP_K`, `PDF_TOP_K`, `EXCEL_TOP_K` in `.env` |
| Document scope | Sidebar `doc_ids` or API |
| Text granularity | `CHUNK_MAX_WORDS`, `CHUNK_OVERLAP_WORDS` |
| LLM context size | `LLM_CONTEXT_MAX_CHARS` |

### Agent mode logs

Grep backend logs for `AGENT`:

```
AGENT turn_start user_id=2 history_turns=3 query_preview='...' max_rounds=1
AGENT groq_request model=openai/gpt-oss-120b message_count=8 tools=4
AGENT tool_start round=1 name=search_documents args={"query": "..."}
AGENT tool_done round=1 name=search_documents chunks=5 images=0 total_chunks=5
AGENT chunks_retrieved tools=['search_documents'] total=5 by_type={'text': 4, 'table': 1}
AGENT answer_stream mode=grounded context_chars=4200 sources=5
```

| Log signal | Likely cause |
|------------|--------------|
| `AGENT route_fallback reason=router_skipped_tools` | Forced `search_documents` ran |
| `AGENT query_rewrite original=... rewritten=...` | Follow-up rewritten before retrieval |
| `AGENT route_clarification` | Router asked which document to use |
| `inline_table_detected=true` | Model jammed pipe tables into prose |

### SQL Agent logs

```
SQL scope_classifier decision=sql confidence=0.95 tables=['film'] reason='...'
SQL route_plan mode=sql tables=['film'] schema_source=memory
SQL schema_cache refreshed user_id=2 connection_id=1 tables=15
```

| Signal | Likely cause |
|--------|--------------|
| `mode=rag` with active SQL | Low classifier confidence or no table match |
| `DetachedInstanceError` on `UserSqlConnection` | Use `snapshot_connection()` before session close |
| Connection test fails from Docker | Use `dvdrental-postgres` hostname, not `localhost` |

### LLM answer quality

```
LLM_CONTEXT query_preview='...' sources=5 context_chars=8200
LLM_ANSWER query_preview='...' answer_chars=420 inline_table_detected=false
```

### Table extraction issues

- Check `extra_metadata.table_qa` on indexed chunks in OpenSearch Dashboards.
- Logs emit `TABLE_QA` lines with misalignment and fallback decisions.
- Whitespace-aligned tables (10-K filings) often trigger geometry/text-line recovery.

### Computed chart issues

```
RETRIEVAL_REQUEST ... table=0 chartable_marked=0 charts_offered=0
```

| Signal | Likely cause |
|--------|--------------|
| `table=0` | No table chunk in top-k |
| `chartable_marked=0` | Table retrieved but no `chart_profile` — re-ingest |
| `validation_outcome: validation_failed` | Stale profile — re-ingest |

### PDF citation viewer

Browser console: `[pdf-viewer]` prefixed logs (`frontend/src/pdf/log.ts`).

| Symptom | Check |
|---------|-------|
| Black screen | Source `page_count` was 0 — viewer uses `pdf.numPages` |
| No highlights | Chunk predates bbox ingestion — **re-upload** |
| `Setting up fake worker failed` | `pdfjs.ts` uses `?worker` + `workerPort` |
| Slow late-page open | Confirm **206** range responses from `/documents/{doc_id}/file` |

### Reindex after changes

Re-upload when changing: embedding model, chunking, chart profile logic, bbox/`line_bboxes` logic, XLSX band/schema settings, DOCX viewer matching.

### Known limitations

- **No vision-model reasoning** — chart Q&A depends on table/text chunks, not pixels.
- **Computed charts require chartable table retrieval** — asking for a "chart" alone does not bypass this (but `create_chart` tool may still run on retrieved tables).
- **English only** — embedding model and OCR tuned for English.
- **DOCX viewer** requires LibreOffice (`LIBREOFFICE_PATH` or auto-detect); soft-fails without preview PDF.
- **PDF highlights require re-ingestion** for documents indexed before bbox support.
- **Highlight scope** — only sources from the **current answer** are highlighted.
- **Answer LLM is single-turn** — chat history goes to router/rewrite only.
- **Agent router** may use multiple Groq calls per message when `AGENT_MAX_ROUNDS` > 1.
- **Async ingestion** via BackgroundTasks — not a durable job queue.
- **In-process rate limits** — per-container; not shared across replicas.
- **SQL from Docker** — user DB on host `localhost` is not reachable; use host gateway or container hostname.

---

## 21. Citation viewers (PDF, DOCX, XLSX)

Citation viewers let users jump from a retrieved source to the exact region in the original document.

### End-to-end flow

```
Query answer completes
       │
       ▼
SSE `sources` event (bbox, line_bboxes, viewer_page, sheet_name, highlight_row, …)
       │
       ▼
User clicks "Open page N in document" / "Open in spreadsheet"
       │
       ├── PDF/DOCX ──► PdfViewerPanel (side drawer)
       └── XLSX ──────► SpreadsheetViewerPanel (full-screen overlay)
```

### PDF citation viewer

#### Ingestion geometry (`backend/app/ingestion/text.py`)

For each PDF **text** chunk:

| Field | Meaning |
|-------|---------|
| `bbox` | Union box `[x0, top, x1, bottom]` in PDF top-origin coords — scroll centering + fallback highlight |
| `extra_metadata.line_bboxes` | One tight box per visual line — **preferred** for rendering |

Lines derived by clustering words with similar `top` values (`_group_lines()`): new line when `|top - current_top| > height × 0.6`. Per-line boxes avoid painting empty gutters between columns that a union box would cover.

**Tables and images** store table/image `bbox` but typically no `line_bboxes` — viewer falls back to union `bbox`.

> **Re-ingestion required:** PDFs uploaded before bbox support open in the viewer but may lack highlights until re-uploaded.

#### Backend PDF serving (`backend/app/api/pdfs.py`)

| Behavior | Detail |
|----------|--------|
| Auth | Bearer required; ownership → **404** |
| Format | PDF only (`find_pdf_path()`) |
| Range | `Range: bytes=start-end` → **206** + `Content-Range` |
| Streaming | 64 KiB chunks; enables PDF.js lazy page fetch |

#### Frontend PDF.js (`frontend/src/pdf/pdfjs.ts`)

- Worker: `pdfjs-dist/build/pdf.worker.min.mjs?worker` → `GlobalWorkerOptions.workerPort`
- `disableAutoFetch: true` — opening page 340 does not download pages 1–339
- `rangeChunkSize: 65536`; authenticated `httpHeaders: { Authorization: Bearer … }`

#### PdfViewerPanel highlight logic

| Feature | Implementation |
|---------|----------------|
| **Box selection** | `line_bboxes` when present; else single `bbox` |
| **Coordinate transform** | PDF top-origin → PDF.js viewport via `convertToViewportPoint(x0, pageHeightPts - top)` |
| **Primary citation** | Clicked source → amber fill `bg-amber-300/25` |
| **Other cited regions** | Same answer, visible pages → sky fill `bg-sky-300/15` |
| **Style** | Fill-only marker pen — **no borders** |
| **Virtualization** | Renders pages within scroll viewport ± `PDF_VIEWER_PAGE_WINDOW` (default 2) |
| **Doc lifecycle** | PDF reloads only when `docId` changes; retargeting another citation scrolls in-place |
| **Highlight scope** | Only `messageSources` from **current answer** — not full document index |
| **Interaction** | Click highlight rect → set as primary + smooth-scroll to page |

**Error isolation:** `PdfViewerBoundary` wraps panel so PDF.js failures don't unmount chat UI.

### DOCX citation viewer

DOCX sources use the **same PDF viewer panel** against a rendered preview PDF.

#### Ingestion pipeline

1. **`render_docx_to_pdf()`** (`docx_render.py`) — LibreOffice headless, timeout **120s**; output `__viewer.pdf` in upload dir. Soft-fails if LibreOffice unavailable.
2. **`locate_chunks_in_viewer_pdf()`** (`docx_bbox_lookup.py`) — mutates chunks in place:
   - Tokenizes chunk content (max **300** tokens) and page words.
   - LCS order-preserving match (tolerates interleaved text).
   - Search window: `[anchor_page, anchor_page + DOCX_VIEWER_SEARCH_WINDOW_PAGES]` (default 3).
   - Success → `viewer_location: { match_status: "ok", viewer_page, bbox, line_bboxes, match_ratio }`.
   - Requires `match_ratio >= DOCX_VIEWER_MIN_MATCH_RATIO` (default **0.6**).

#### UI behavior

- Sources show **section name** (e.g. `Lists`) or `part N`, not `p. N`.
- `viewer_page` used instead of `page_number` for PDF.js navigation.
- `source_format: "docx"` in sources SSE payload.

### XLSX spreadsheet viewer

**Component:** `SpreadsheetViewerPanel.tsx` + `SpreadsheetGrid`

| Feature | Behavior |
|---------|----------|
| **Data loading** | `getSpreadsheetMetadata` → sheet tabs; `getSpreadsheetSheet(docId, sheet)` → grid |
| **Initial sheet** | `target.sheetName` → `source.sheet_name` → index match → first sheet |
| **Highlight eligibility** | Only when `sheet_role === "primary"` AND active sheet matches source |
| **Row highlight** | `highlight_row` from post-answer entity matching |
| **Row range** | Span ≤ **24** rows → full highlight band; wider → scroll only |
| **Satellite/standalone** | Data viewable; no row highlight band |

Post-answer: `apply_xlsx_highlights_to_sources()` in `xlsx_highlight.py` sets `highlight_row` on sources.

### Configuration

| Setting | Default | Purpose |
|---------|---------|---------|
| `PDF_VIEWER_PAGE_WINDOW` | 2 | Extra pages rendered above/below scroll viewport |
| `DOCX_VIEWER_SEARCH_WINDOW_PAGES` | 3 | Forward search window for bbox lookup |
| `DOCX_VIEWER_MIN_MATCH_RATIO` | 0.6 | Min token match ratio to accept highlight |

Exposed to frontend via `GET /config` as `pdf_viewer_page_window`.

### Tests

- `backend/tests/test_pdf_serving.py` — range parsing and 206 responses
- DOCX viewer/bbox tests in `backend/tests/test_docx_*.py`
- XLSX highlight tests in `backend/tests/test_xlsx_highlight*.py`

---

## 22. Request logging

**Location:** `backend/app/retrieval/request_log.py`

Every `/query/stream` and `/search` request logs structured lines for debugging retrieval and answer quality.

### Log lines

| Tag | Content |
|-----|---------|
| `RETRIEVAL_REQUEST` | Compact: retrieved count, table count, chartable marked, charts offered, chunk type breakdown |
| `RETRIEVAL_REQUEST_DETAIL` | Full JSON: per-chunk summary + `chart_eligibility` (`validation_outcome`: `offered`, `validation_failed`, `not_marked_at_ingestion`, etc.) |
| `LLM_CONTEXT` | `query_preview`, `sources`, `context_chars` |
| `LLM_CONTEXT_DETAIL` | Full context string sent to answer LLM |
| `LLM_ANSWER` | `answer_chars`, `inline_table_detected` |
| `LLM_ANSWER_DETAIL` | Full answer text |
| `LLM_ANSWER_INLINE_TABLE` | Warning when model embedded pipe tables in prose |
| `QUERY_STREAM_DONE` | `ok=true/false` on stream completion |
| `TABLE_QA` | Per-table extraction QA during ingestion |
| `XLSX_HIGHLIGHT` | Post-answer highlight application stats |

### Agent / SQL tags

Grep for `AGENT`, `SQL route_plan`, `SQL scope_classifier`, `SQL schema_cache` in backend logs during `/query/stream`.

---

## Quick reference: query lifecycle

```
User message
     │
     ├─► Query rewrite (aux LLM, optional)
     │
     ├─► SQL route plan (if active connection)
     │        ├─ sql ──────► LangChain SQL agent ──► tokens + sql_meta
     │        ├─ hybrid ───► SQL then RAG
     │        └─ rag/agent_only ──► skip SQL
     │
     ├─► RAG agent (Groq tools, sql_active=False)
     │        └─► hybrid_retrieve / search_images / create_chart / list_documents
     │
     ├─► stream_groq_answer (grounded excerpts + sql_context_note)
     │
     └─► SSE: sources, charts, persist message (sources, charts, sql_meta)
```

```
Upload (PDF / DOCX / XLSX)
     │
     ▼
extract → embed → OpenSearch (user-scoped)
     │
     ▼
hybrid_search ← embed(query)     [via agent tools or direct /search]
     │
     ├─ text/table → LLM context
     ├─ image → Sources + hero (not LLM)
     ├─ chartable table → chart spec / QuickChart
     └─ xlsx → row highlights in viewer
```
