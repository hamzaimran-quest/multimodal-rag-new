# Multimodal RAG System — Architecture Reference

This document is the single source of truth for the system design. Update it whenever
an implementation decision deviates from what's written here, so later phases stay
consistent with what was actually built (not just what was planned).

---

## 1. System Overview

A multimodal RAG system that ingests PDFs (text, tables, images/charts), indexes them
in OpenSearch using hybrid search (BM25 + vector), and answers queries with streamed
responses and cited sources (filename, page number, chunk type).

```
[Settings UI] --upload--> [Ingestion Pipeline] --> [OpenSearch]
                                                         ^
[Chat UI] --query--> [Query API] --hybrid search---------|
              <--stream response + sources--
```

### Core components
1. OpenSearch (storage + hybrid search engine)
2. Ingestion pipeline (PDF → chunks → embeddings → indexed docs)
3. Query API (hybrid retrieval → LLM → streamed response + sources)
4. Frontend: Settings tab (doc management) + Chat tab (query interface)

---

## 2. Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Search/vector store | OpenSearch (Docker) | k-NN plugin enabled, hybrid search pipeline |
| Backend API | FastAPI | async support, plays well with streaming |
| Embeddings | FastEmbed (ONNX) — `all-MiniLM-L6-v2` (384-dim) | local model, no PyTorch; explicit L2 normalization |
| Text/table extraction | pdfplumber / camelot | text-layer first, positional-aware |
| OCR (scanned text/tables) | Tesseract or PaddleOCR | CPU-only, free, local — for scanned/image-only pages that are plain text or simple tables |
| Vision fallback (charts/graphs) | **Deferred / not in MVP scope.** Charts are retrieved and displayed via proximity-based context (surrounding text + OCR keyword signal), not vision-model reasoning. See sections 4 & 6 for the image-retrieval approach. Revisit a hosted API (e.g. Gemini free tier) only if true chart-value Q&A becomes a requirement later. |
| LLM (generation) | Groq Cloud API — `llama-3.3-70b-versatile` | free tier; strong reasoning quality for financial Q&A over tables |
| Streaming | Server-Sent Events (SSE) | simplest one-directional token streaming |
| Frontend | Vite + React + Tailwind | fast to scaffold, no heavy component library overhead for MVP |

---

## 3. OpenSearch Setup

### docker-compose services
- `opensearch` (single-node, dev mode, security plugin disabled for local dev)
- `opensearch-dashboards` (optional, for manual index inspection/debugging)
- `backend` (FastAPI app — ingestion + query API)

### Key settings
- k-NN plugin enabled: `index.knn: true`
- Single-node discovery type for dev: `discovery.type=single-node`
- Disable security plugin for local dev only: `plugins.security.disabled=true`

---

## 4. Index Design

**One index**, one document per chunk (text, table, or image-derived). This is the
central data model — every other component depends on this schema staying consistent.

### Document schema

```json
{
  "chunk_id": "uuid",
  "doc_id": "uuid",
  "filename": "huawei_2025.pdf",
  "page_number": 12,
  "chunk_type": "text | table | image",
  "content": "raw or markdown text of the chunk (this is what gets embedded + BM25-matched)",
  "embedding": [0.123, "..."],
  "bbox": [0, 0, 0, 0],
  "image_path": "data/images/{doc_id}/page12_img0.png",
  "upload_timestamp": "ISO-8601 string",
  "extra_metadata": {
    "table_headers": ["..."],
    "image_caption": "...",
    "extraction_method": "pdfplumber | camelot | ocr_proxy | vision_model"
  }
}
```

### Image chunks — no vision-model reasoning for MVP
For `chunk_type: "image"`, the `content` field is **not** a vision-model-generated
caption. Instead it's built from proximity signals so the chunk is still retrievable
via hybrid search without any vision reasoning:
- Nearby page text (section heading, preceding/following paragraph)
- Raw OCR output from the image itself (Tesseract/PaddleOCR) — even if the numbers
  come out unordered/scrambled relative to their visual position (see the Huawei
  chart example in prior notes), the raw text is still useful *keyword* signal for
  retrieval (e.g. "Revenue," "CNY Million," axis year labels)

`image_path` stores where the extracted image file lives on disk so it can be served
back to the frontend and rendered inline in the Chat UI when this chunk is retrieved
as a relevant result — the image is shown because it's contextually relevant, not
because the LLM reasoned about its content. `extraction_method: "ocr_proxy"` marks
chunks built this way, distinct from a true future vision-model pass.

### Document-level status tracking
Since ingestion is async (see section 9 / Open Decisions), track status per `doc_id`
separately from chunk documents — e.g., a lightweight status doc (`{"doc_id": "...",
"filename": "...", "ingestion_status": "pending | processing | indexed | failed",
"upload_timestamp": "..."}`) in the same or a separate index. This is what powers the
"list documents" view and upload progress in the Settings UI.


- `doc_id`: parent document identifier. **This is the delete-cascade key** — deleting
  a document = one `delete_by_query` on `doc_id`, removing all its chunks regardless
  of type. Do not skip this field or derive it inconsistently.
- `chunk_type`: always one of `text`, `table`, `image`. Used for filtering/debugging
  and for deciding how to render the source citation in the UI.
- `embedding`: `knn_vector` field type. Dimension must match the sentence-transformer
  model in use (e.g., 384 for `all-MiniLM-L6-v2`). **If you change embedding models
  later, the index must be recreated** — dimension mismatches will break the mapping.
- `bbox`: optional but recommended if you want to highlight the exact source location
  on a page image in the UI later. Can be deferred to a later phase.
- `content`: this is the human-readable form of the chunk. For tables, store as
  markdown-formatted table text (not raw JSON) so it reads naturally both for BM25
  matching and for feeding into the LLM context at query time.

### Index mapping requirements
- `content`: `text` type, standard analyzer (BM25 matching)
- `embedding`: `knn_vector` type, dimension = embedding model output size
- `doc_id`, `chunk_type`, `filename`: `keyword` type (exact-match filtering/aggregation)
- `page_number`: `integer`
- `upload_timestamp`: `date`

---

## 5. Hybrid Search

Use OpenSearch's built-in **hybrid query + search pipeline** (normalization processor)
rather than hand-rolling score fusion.

- Search pipeline: `normalization-processor` (min-max or L2 norm) blending BM25 score
  and k-NN score.
- Query API sends a single hybrid query with two clauses:
  - `match` clause on `content` (BM25)
  - `knn` clause on `embedding` (vector similarity)
- Weights between BM25 and vector score are configurable in the pipeline definition —
  start with equal weighting (0.5/0.5) and tune later based on retrieval quality.

---

## 6. Ingestion Pipeline

### Flow (per uploaded PDF)
1. Generate a `doc_id` for the file.
2. Parse page by page. Route content by type:
   - **Text blocks** → direct text extraction, chunked by paragraph/section
     (~300–500 tokens, with overlap).
   - **Tables** → pdfplumber/camelot first (try "lattice" mode for bordered tables,
     "stream" mode for whitespace-aligned tables). If extraction confidence is low
     or the page has no text layer, fall back to a vision-model reconstruction pass.
     Store the whole table as **one chunk** — do not split rows across chunks.
   - **Images/charts (MVP approach — no vision-model reasoning)**:
     - Extract the image and save it to disk (`data/images/{doc_id}/page{N}_img{i}.png`),
       recording the path in `image_path`.
     - Run OCR (Tesseract/PaddleOCR) on the image to pull out any raw text signal
       (numbers, labels, axis text) — this is used purely as retrieval keyword
       signal, not for reconstructing chart structure. Combine this with nearby page
       text (section heading, preceding/following paragraph) to build the chunk's
       `content` field.
     - This makes the image chunk **retrievable via hybrid search** (e.g. a query
       like "financial highlights" matches the section heading + OCR keywords near
       a chart) **without any vision-model call**. When such a chunk is retrieved,
       the Chat UI renders the actual image inline alongside the streamed answer —
       the image is shown because it's contextually relevant, not because the LLM
       reasoned about its visual content.
     - Skip purely decorative images (heuristic: very small images, or images with
       no nearby text signal and no meaningful OCR output, can be flagged for
       skip — refine this heuristic once you see real false positives/negatives).
     - `extraction_method: "ocr_proxy"` for chunks built this way. True vision-model
       chart-value reasoning (Gemini free tier or similar) is **deferred and out of
       MVP scope** — only revisit if chart-specific numeric Q&A becomes a real
       requirement later; nothing built here needs to be thrown away if you do.
3. Embed each chunk's `content` field using FastEmbed (`all-MiniLM-L6-v2`, L2-normalized).
4. Write each chunk to OpenSearch with full metadata (`doc_id`, `filename`,
   `page_number`, `chunk_type`, `extraction_method`, etc.).
5. Return `doc_id` + ingestion status to the caller once complete.

### Known extraction risks (from prior testing — see chunk_type handling above)
- Charts often render as an image **plus** a scrambled text layer with the underlying
  numbers, out of visual/logical order (numbers present, but disconnected from which
  category/year they belong to). Do not rely on raw text-layer order for chart data —
  use vision-model reconstruction for these, or explicit coordinate-based parsing.
- Multi-page tables: headers may or may not repeat per page. Detect continuation
  (no new header row, same column structure as previous page's table) and stitch
  before chunking, rather than treating each page's fragment as a separate chunk.
- Some PDFs may contain leftover non-content text (e.g., embedded comments/annotation
  layers) that should be filtered out before embedding — flag and log anomalous
  short/out-of-place text fragments during ingestion for manual review.

---

## 7. Query API

### Flow
1. Receive user query, plus optional `doc_id` filter (user may scope search to a
   single uploaded document instead of the full corpus — support this from the start).
2. Embed query using the same sentence-transformer model used at ingestion time
   (`all-MiniLM-L6-v2`).
3. Run hybrid search (BM25 + k-NN) against the index → top **8** chunks with metadata
   (default `top_k = 8`; expose as a tunable query param for later experimentation).
   If a `doc_id` filter is present, apply it as a filter clause alongside the hybrid
   query.
4. Construct LLM context from retrieved chunks' `content` field.
5. Call Groq Cloud API (`llama-3.3-70b-versatile`) with a system prompt that enforces
   **strict grounding**: answer only from the provided chunk content, and explicitly
   respond with something like "not found in the provided documents" if the requested
   figure/fact isn't present in the retrieved context. This matters especially for
   financial figures, where a confident-sounding hallucinated number is worse than an
   honest "I don't have that."
6. Stream the LLM's generated response token-by-token via SSE. The LLM's text
   answer is grounded only in `text`/`table` chunk content (see grounding requirement
   above) — image chunks are not fed to the LLM as reasoning input, only used for
   display (see step 7).
7. Attach source metadata (filename, page number, chunk_type) as a **structured
   payload sent after the stream completes** (confirmed MVP approach — see Source
   citation requirement below for why inline citation is deferred). **If any
   retrieved chunk has `chunk_type: "image"`, include its `image_path` in this
   payload** so the Chat UI can render the image inline alongside the answer —
   this is how charts (e.g. the Huawei Five-Year Financial Highlights bar charts)
   surface in results without any vision-model involvement.

### Source citation requirement
Every response must be traceable back to specific chunks. Minimum metadata to surface
in the UI per source: `filename`, `page_number`, `chunk_type`. **Show a snippet of the
retrieved `content` on hover/expand from day one** (not deferred) — this is central to
user trust in a financial-data tool and is a small addition once `content` is already
stored per chunk.

**Citation UX decision**: use the preamble/final-payload pattern, not inline
chunk-ID citation. Inline citation-by-chunk-ID is unreliable with smaller/faster
models like Groq's Llama models — they tend to hallucinate or skip chunk IDs.
Revisit inline citation later once the core pipeline is proven, if desired.

---

## 8. Settings Tab (Document Management)

### Upload
- POST file → backend generates `doc_id`, kicks off ingestion via FastAPI
  `BackgroundTasks`, and returns immediately with `doc_id` + `status: "processing"`.
  Frontend polls the document-registry status (see section 4) until it flips to
  `indexed` or `failed`.

### List documents
- Aggregation query on the index grouping by `doc_id` (or `filename`), returning one
  row per document (not per chunk) — show filename, upload date, chunk count,
  ingestion status.

### Delete
- `delete_by_query` with `{"term": {"doc_id": "<id>"}}` — removes every chunk
  (text, table, image) associated with that document in a single call.
- Confirm deletion completed (check deleted count matches expected chunk count)
  before showing success in the UI.

---

## 9. Build Order (phases)

1. **OpenSearch + index mapping** — docker-compose up, create index with the schema
   above, verify k-NN works with dummy vectors (no ingestion logic yet).
2. **Ingestion: text + native tables only** — skip vision-model image handling
   initially. Get chunks reliably written to OpenSearch with correct metadata.
3. **Hybrid search retrieval** — verify BM25 + k-NN fusion returns sensible results
   via direct API/curl calls (no UI yet).
4. **Settings UI** — upload / list / delete, wired to the ingestion pipeline and
   index from phases 1–2.
5. **Query API with streaming + source citation** — SSE streaming, sources attached.
6. **Chat UI** — connect to the streaming query API, render citations.
7. **Vision-model image/chart extraction** — layer in last, once the core
   text/table pipeline is proven end-to-end.

Do not skip ahead to a later phase until the current one is verified — check both
that it runs without errors AND that the output is actually correct (e.g., inspect
real indexed chunks, not just confirm the API returns 200).

---

## 10. Open Decisions / To Be Filled In As Built

Use this section to track decisions made during implementation that aren't finalized
in this doc yet. Update as Cursor builds each phase.

- [x] **Sentence-transformer model: `all-MiniLM-L6-v2` (384-dim) — final choice.**
      Note: an earlier pass in this doc recommended `bge-small-en-v1.5` for slightly
      better retrieval quality at similar speed; user confirmed `all-MiniLM-L6-v2`
      instead. This is fine — the quality gap is small, and MiniLM is simpler/more
      widely documented. Revisit only if retrieval quality proves insufficient in
      testing.
- [ ] Chunk size / overlap parameters for text chunking.
- [ ] OCR tool choice for scanned text/simple tables (Tesseract vs. PaddleOCR).
- [x] Vision model for chart/graph extraction: **out of MVP scope.** Charts are
      retrieved and displayed via proximity-based context (OCR keyword signal +
      nearby page text), not vision-model reasoning — see sections 4 & 6. A hosted
      API (Gemini free tier) remains the fallback plan only if true chart-value
      Q&A becomes a real requirement later.
- [ ] Heuristic for classifying an image as "decorative / skip" vs. "index via
      OCR + proximity" during ingestion.
- [ ] Static file serving for extracted images: how does the frontend fetch
      `image_path` for display (e.g. a FastAPI static files route serving
      `data/images/`, or returning base64 in the API response instead of a path)?
- [x] LLM provider/model: **Groq Cloud API — `llama-3.3-70b-versatile`**. Free tier;
      chosen for strong reasoning quality on financial Q&A over tables. Confirm Groq
      API key is set up (env var) as part of Phase 1 scaffolding, and be aware of
      free-tier rate limits if query volume grows during dev/testing.
- [x] Async ingestion: **upload returns immediately** with `doc_id` + `status: "processing"`;
      ingestion runs in the background (FastAPI `BackgroundTasks` sufficient at dev
      scale — revisit a real task queue like Celery/RQ only if this becomes a
      bottleneck). Track `ingestion_status` (`pending / processing / indexed / failed`)
      per `doc_id` so the Settings UI can show progress and the "list documents"
      view has real status to display.
- [x] Ingestion status storage: **document-registry doc type within OpenSearch**
      (not a separate SQLite/JSON store) — keeps infra to one system rather than two.
- [x] PDF source assumption: design for a **mix of native text-layer and scanned
      pages** — don't special-case only one.
- [x] Numeric accuracy requirement: **strict grounding**. LLM must answer only from
      retrieved chunk content and explicitly state when a figure/fact isn't found in
      the provided documents, rather than guessing — see section 7.
- [x] Default retrieval **top_k = 8** (tunable query param); **doc-level filtering
      supported** from the start via an optional `doc_id` param on the query API.
- [x] Citation UX: **preamble/final structured payload after the stream completes**,
      not inline chunk-ID citation (unreliable with smaller/faster Groq models).
      Source snippets shown on hover/expand from day one.
- [x] Frontend stack: **Vite + React + Tailwind**.
- [x] Environment/deployment: **local-only dev** for now (Docker Desktop or
      equivalent) — no auth, CORS hardening, or secrets management needed yet;
      revisit if/when deployed beyond local use.
- [x] Auth/multi-user: **single-user, local tool** — no access control on the
      Settings tab for now.
- [x] Sample validation data: **Huawei 2025 Annual Report** (clean IFRS-style report,
      native text-layer, charts with scrambled text-layer values — good for chart
      extraction testing) and **Timberland Bancorp 2025 10-K** (dense nested
      financial tables, whitespace-aligned table stress test, some extraction
      noise/artifacts already observed) — use both across phases for validation.
- [x] Language: **English-only** for now (corpus and queries). Revisit OCR/embedding
      model choice if multilingual filings are added later.