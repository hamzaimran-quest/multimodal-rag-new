# SQL Agent Feature — Implementation Plan

This document describes how to add an optional **SQL Agent** alongside the existing **RAG (document) agent**, without breaking current behavior when no database is connected.

> **Stack note:** The linked tutorial is [LangChain JS SQL agent](https://docs.langchain.com/oss/javascript/langchain/sql-agent). This project’s backend is **Python (FastAPI)**. Implementation uses the **current Python LangChain SQL stack** (pinned at implementation time — see §5.3), not legacy `langchain` 0.1 APIs:
>
> - `langchain` + `langchain-community` — `SQLDatabase`, `SQLDatabaseToolkit`, `create_sql_agent`
> - `langchain-groq` — `ChatGroq` pointed at **Groq Cloud** (`GROQ_API_KEY`, same as the rest of the app)
>
> Built-in SQL toolkit tools only (`sql_db_list_tables`, `sql_db_schema`, `sql_db_query`, etc.) — no custom SQL tool implementations.

---

## 1. Goals

| Goal | Detail |
|------|--------|
| Optional SQL capability | Per-user **saved** PostgreSQL connections; **one active** at a time; when none active, app behaves exactly as today (RAG only). |
| Unified chat | One composer; user asks document questions, SQL questions, or hybrid — router picks the right path. |
| User-provided context | On connect, user supplies a **database description** (domain, key tables, business rules) injected into the SQL agent prompt. |
| Visible connection state | Sidebar shows **SQL Agent** with ✓ connected / ✗ not connected. |
| Security-first | Treat model-generated SQL as untrusted; narrow DB permissions, encrypt secrets, audit queries. |

---

## 2. Non-goals (v1)

- **PostgreSQL only** for user databases (no MySQL, SQLite, or other dialects in v1).
- Replacing the existing Groq document router with LangChain for RAG.
- Writing custom SQL tools (use LangChain’s SQL toolkit only).
- A second Groq “rephrase/polish” pass on SQL answers (stream LangChain output directly).
- Letting the SQL agent write to the **app’s own Postgres** (users/chats) — that DB stays internal.

---

## 3. High-level architecture

```mermaid
flowchart TD
    UI[Chat UI + Sidebar] --> API[FastAPI]
    API --> Conn[SQL Connection API]
    API --> Query["/query/stream"]

    Conn --> AppDB[(App Postgres<br/>users, chats, sql_connections)]

    Query --> Orchestrator[Top-level Router]
    Orchestrator -->|document intent| RAG[Groq RAG Agent<br/>existing tools]
    Orchestrator -->|SQL intent| SQL[LangChain SQL Agent<br/>built-in SQL tools]
    Orchestrator -->|hybrid| Both[SQL then RAG<br/>sequential merge]

    SQL --> UserDB[(User PostgreSQL<br/>read-only)]
    RAG --> OS[OpenSearch + Groq answer]
```

**Principle:** RAG pipeline (`search_documents`, `search_images`, `create_chart`, etc.) stays unchanged. SQL is a **parallel capability** gated on **having an active connection**.

---

## 3.1 Locked-in product decisions

| Decision | Choice |
|----------|--------|
| Saved connections | **Multiple per user**, each with its own description |
| Active connection | **Exactly one active** at a time; others saved but deactivated |
| Dialect | **PostgreSQL only** |
| Hybrid merge | **Sequential:** SQL agent first → RAG retrieval/answer second, tokens appended in one assistant message |
| SQL answer LLM | **LangChain SQL agent only** (Groq via `ChatGroq`) — **no** extra Groq polish pass |
| Streaming | Same SSE `token` events as today; SQL agent streams through Groq Cloud inside LangChain |
| LangChain version | Current compatible packages (pin in `requirements.txt` at implementation; avoid deprecated 0.1 agent APIs) |

## 4. User experience

### 4.1 Sidebar

Add a nav row under **Documents**:

```
SQL Agent          ✓   (green tick when an active connection exists)
                   ✗   (muted cross when no active connection)
```

- **Click** → open **SQL Agent** panel: list saved connections, add new, activate another, edit description, test, delete.
- Sidebar tick/cross reflects **active connection only** (not “any saved row exists”).
- Status refreshes on load and after connect / activate / disconnect / delete.

### 4.2 Connection management (multi-save, single active)

**Add connection** form:

| Field | Required | Notes |
|-------|----------|-------|
| Connection string | Yes | `postgresql://user:pass@host:5432/dbname` or `postgresql+psycopg2://...` |
| Database description | Yes | Free text: “E-commerce DB. `orders` has line items; `customers` is PII…” |
| Display name | Yes | Short label, e.g. “Production analytics”, “Staging warehouse” |

**Add flow:**

1. User submits → backend validates **PostgreSQL** URL only (reject other schemes).
2. Backend runs **test query** (`SELECT 1`) with timeout.
3. On success → encrypt URL, insert row with `is_active=false` by default.
4. If this is the user’s **first** saved connection → set `is_active=true` automatically.
5. On failure → safe error (no password in message).

**Saved connections list** (panel):

- Each row: display name, description snippet, active badge, **Activate**, **Edit description**, **Test**, **Delete**.
- **Activate** → sets `is_active=true` on chosen row, `is_active=false` on all other rows for that user (transaction).
- **Delete** → remove row; if deleted row was active, clear active state (sidebar ✗) unless another connection is promoted.
- **Deactivate all** (optional explicit action) → no active connection; SQL tools disabled; RAG only.

**Disconnect** in v1 means **deactivate all** or **delete active** — not “wipe all saved connections” unless user deletes each one.

### 4.3 Chat behavior

| Connection state | Behavior |
|------------------|----------|
| No active connection | Only RAG agent active (current behavior). Saved-but-inactive rows do not enable SQL. |
| Active connection set | Router may invoke **SQL agent**, **RAG agent**, or **both** (sequential) per message. |

**Examples:**

| User message | Expected route |
|--------------|----------------|
| “What is our Q3 revenue?” (DB has sales) | SQL only |
| “Summarize the chairwoman message” (scoped PDF) | RAG only |
| “Compare headcount in the database with the table in demo.docx” | **SQL first** (stream SQL answer tokens) → **then RAG** (stream document answer tokens into same message) |
| “Hello” | Direct reply, no tools |

### 4.4 Answer presentation & streaming

- **RAG-only:** unchanged — Groq `stream_groq_answer` → SSE `token` events.
- **SQL-only:** LangChain SQL agent with **`ChatGroq(streaming=True)`** → map LLM token callbacks / `astream_events` → same SSE **`token`** events (no second Groq call).
- **Hybrid (sequential):** stream SQL agent tokens first; when SQL phase completes, stream RAG answer tokens immediately after in the **same** assistant bubble (one message, one token stream).
- **SQL provenance:** after SQL phase, emit SSE **`sql`** with executed queries (collapsible “SQL used” in UI).
- **RAG artifacts:** `sources` / `charts` events unchanged, emitted during RAG phase only.

---

## 5. Backend design

### 5.1 New persistence: `user_sql_connections`

Store in **app Postgres** (not OpenSearch). **Many rows per user**; **at most one** with `is_active=true` (enforce in application code + partial unique index).

| Column | Type | Purpose |
|--------|------|---------|
| `id` | PK | Connection id |
| `user_id` | FK, indexed | Owner |
| `display_name` | string | UI label |
| `description` | text | User’s DB description for the agent |
| `connection_url_encrypted` | text | Fernet-encrypted PostgreSQL URL |
| `dialect` | string | Always `postgresql` in v1 |
| `is_active` | boolean | Only one `true` per user |
| `created_at` | timestamp | |
| `last_tested_at` | timestamp | |
| `last_error` | text, nullable | Sanitized last failure |

**Constraint:** `UNIQUE (user_id) WHERE is_active = true` (partial unique index) or equivalent transactional activate pattern.

**Never** store plaintext passwords in logs, chat history, or API responses.

**Encryption:** derive key from `JWT_SECRET` or dedicated `SQL_CREDENTIALS_KEY` env var (document in `.env.example`).

### 5.2 New API routes (`/sql-agent`)

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `GET` | `/sql-agent/status` | user | `{ has_active, active_connection?, connections: [{ id, display_name, description, is_active, last_tested_at }] }` — no secrets |
| `GET` | `/sql-agent/connections` | user | List all saved connections (metadata only) |
| `POST` | `/sql-agent/connections` | user | Add: `{ connection_url, description, display_name, activate?: bool }` → test + save |
| `POST` | `/sql-agent/connections/{id}/activate` | user | Set active (deactivates others) |
| `POST` | `/sql-agent/connections/{id}/test` | user | Test saved connection by id |
| `PATCH` | `/sql-agent/connections/{id}` | user | Update `description` and/or `display_name` |
| `PATCH` | `/sql-agent/connections/{id}/credentials` | user | Replace connection URL (re-test required) |
| `DELETE` | `/sql-agent/connections/{id}` | user | Delete one saved connection |
| `POST` | `/sql-agent/deactivate` | user | Clear active flag on all connections (RAG-only mode) |

Rate-limit connect/credential attempts (reuse query rate limit pattern).

### 5.3 LangChain SQL agent module

**New package:** `backend/app/sql_agent/`

```
sql_agent/
  __init__.py
  connection.py      # load active connection, decrypt URL, SQLAlchemy engine cache (TTL)
  agent.py           # create_sql_agent + streaming invoke
  prompts.py         # system prompt: active connection description + safety rules
  streaming.py       # bridge LangChain/Groq tokens → async token iterator
  models.py          # SqlAgentResult dataclass
```

**Dependencies** (`requirements.txt`) — pin **compatible current** versions at implementation (verify against Python 3.12):

```text
langchain>=0.3,<2
langchain-community>=0.3,<2
langchain-groq>=0.2,<1
```

Use imports from current docs, e.g.:

- `langchain_community.utilities.SQLDatabase`
- `langchain_community.agent_toolkits.sql.toolkit.SQLDatabaseToolkit`
- `langchain_community.agent_toolkits.create_sql_agent`
- `langchain_groq.ChatGroq`

**LLM:** Groq Cloud via LangChain (same API key as the app):

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model=settings.sql_agent_model,      # e.g. openai/gpt-oss-20b or groq_aux_model
    groq_api_key=settings.groq_api_key,
    temperature=0,
    streaming=True,
)
```

**Agent construction:**

```python
db = SQLDatabase.from_uri(
    postgresql_url,
    sample_rows_in_table_info=3,
)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm,
    toolkit=toolkit,
    agent_type="tool-calling",
    verbose=False,
    max_iterations=settings.sql_agent_max_steps,
    agent_executor_kwargs={"return_intermediate_steps": True},
)
```

Built-in toolkit tools only — do not reimplement `sql_db_query` etc.

**User description injection:** prepend to SQL agent system prompt (from **active** connection row):

> “Database context (provided by the user): {description}. Use this to interpret table/column names and business meaning.”

**Streaming implementation:**

- Prefer `agent.astream_events(version="v2")` (or equivalent supported API) and yield `on_chat_model_stream` token deltas.
- Fallback: `AsyncCallbackHandler` on `ChatGroq` streaming callbacks.
- Reuse existing `_sse("token", {"text": chunk})` in `query.py` so the frontend requires **no new stream protocol** for SQL text.

**Execution limits:**

- Max agent iterations (`SQL_AGENT_MAX_STEPS`, e.g. 10)
- Per-query timeout (e.g. 30s)
- Max rows returned (e.g. 100) — truncate with note in streamed answer
- PostgreSQL URLs only; normalize to `postgresql+psycopg2://` for SQLAlchemy

### 5.4 Orchestration: extend query pipeline

Today: `POST /query/stream` → `iter_agent_turn` (Groq RAG router) → `stream_groq_answer` → SSE `token`.

**Flow when user has an active SQL connection:**

```
1. Load active user_sql_connections row (is_active=true); if none → today’s RAG-only path
2. Groq router (existing) with extra tool query_database when active
3. Execute branch:
   - RAG-only  → existing iter_agent_turn + stream_groq_answer
   - SQL-only  → stream_sql_agent_turn (LangChain + ChatGroq) → SSE token only
   - Hybrid    → sequential:
       a. stream_sql_agent_turn → tokens + sql SSE
       b. iter_agent_turn (RAG) with SQL result injected as prior context note
       c. stream_groq_answer → append tokens to same assistant message
4. Persist message: content, sources, charts, sql_queries, route_mode
```

**No repolish pass:** SQL text is whatever the LangChain agent streams via Groq; do not pipe it through `stream_groq_answer`.

**Hybrid sequential detail:** after SQL streaming completes, pass a short **internal context block** to the RAG answer step (not shown as a separate user message), e.g.:

> “Database query result (already shown to user): …”

Then RAG retrieval + grounded answer streams as today.

**Router tool `query_database`:** registered **only when `is_active` connection exists**. Router prompt:

- `query_database` — live **active** PostgreSQL facts.
- `search_documents` — uploaded files.
- **Both** when the question compares or combines them (SQL phase runs first).
- If no active connection, tool is omitted — router cannot call it.

Reuses existing multi-round loop in `agent.py` where possible; SQL execution may be a dedicated executor called from `execute_query_database`.

### 5.5 SSE events

Existing: `meta`, `tool`, `token`, `sources`, `charts`, `error`, `done`.

| Event | Payload | When |
|-------|---------|------|
| `token` | `{ "text": "..." }` | **SQL and RAG** answer text (same as today) |
| `sql` | `{ connection_id, display_name, queries: string[], tables?: string[] }` | After SQL agent phase completes |
| `route` | `{ mode: "rag" \| "sql" \| "hybrid" }` | Optional UI badge |

During SQL agent tool steps, emit existing **`tool`** events with labels like “Querying database…” (mirrors RAG tool UX).

Frontend: no new stream parser — append all `token` events to the current assistant message; store `sql` for provenance panel.

### 5.6 Chat history

Extend `ChatMessage` JSON (or add column):

```json
{
  "sql_queries": ["SELECT COUNT(*) FROM orders WHERE ..."],
  "sql_mode": "sql"
}
```

Reloaded chats show SQL provenance the same as live streams.

---

## 6. Frontend design

### 6.1 Types & API client

- `SqlConnection`, `SqlAgentStatus` in `frontend/src/types.ts`
- `frontend/src/api/sqlAgent.ts` — list, add, activate, deactivate, test, patch, delete

### 6.2 Sidebar (`App.tsx`)

- Fetch `GET /sql-agent/status` on mount and after connection changes
- **SQL Agent** row: ✓ if `has_active`, else ✗
- New view: `View = "chat" | "docs" | "sql"`

### 6.3 SQL Agent panel

- **Saved connections** list with active badge
- **Add connection** form (URL, display name, description, optional “Set active”)
- Per row: **Activate**, **Test**, **Edit**, **Delete**
- **Deactivate all** → RAG-only mode without deleting saved rows
- Security note: “Use a read-only PostgreSQL user.”

### 6.4 Chat UI

- No mode toggle required if router handles intent (user’s request).
- Optional small badge on assistant messages: “Answered from database” / “From documents” / “Combined”.
- `SqlProvenancePanel` (collapsible): executed SQL, similar styling to Sources panel.

---

## 7. Security & operations

| Risk | Mitigation |
|------|------------|
| Arbitrary SQL execution | DB user: `SELECT` only; no `INSERT/UPDATE/DELETE/DDL` |
| Credential leak | Encrypt at rest; never return URL in GET; redact logs |
| Cross-user access | All endpoints scoped by `user_id` from JWT |
| App DB vs user DB | Separate engines; SQL agent never points at `DATABASE_URL` |
| DoS / heavy queries | Timeouts, row limits, rate limits, connection pool caps |
| Prompt injection via description | Treat description as untrusted hint; don’t bypass SQL toolkit |
| Network | User DB must be reachable from backend container (document VPN/tunnel needs) |

**Docker note:** if user DB is on `localhost` from their machine, backend in Docker cannot reach it — document use of host gateway or cloud DB URL.

---

## 8. Configuration

New env vars (`.env.example`):

```env
# SQL Agent (optional user PostgreSQL connections)
SQL_AGENT_ENABLED=true
SQL_AGENT_MODEL=openai/gpt-oss-20b          # ChatGroq model for LangChain SQL agent
SQL_AGENT_MAX_STEPS=10
SQL_AGENT_QUERY_TIMEOUT_SECONDS=30
SQL_AGENT_MAX_ROWS=100
SQL_CREDENTIALS_KEY=                        # optional; defaults to derived from JWT_SECRET
SQL_CONNECTION_CACHE_TTL_SECONDS=300
```

Uses existing `GROQ_API_KEY` (Groq Cloud). Default `SQL_AGENT_MODEL` may alias `GROQ_AUX_MODEL` if unset.

---

## 9. Implementation phases

### Phase 1 — Connection plumbing (no chat changes)
- DB migration: `user_sql_connections` (multi-row, `is_active`)
- Encrypt/decrypt helpers; PostgreSQL URL validation only
- `/sql-agent/*` CRUD, activate, deactivate
- Sidebar status + SQL Agent panel (list, add, activate)
- Tests: multiple saves, single active invariant, auth isolation

### Phase 2 — LangChain SQL agent (isolated)
- `sql_agent/agent.py` + `streaming.py` with `ChatGroq(streaming=True)`
- Debug-only `POST /sql-agent/ask` to verify streaming + intermediate SQL capture
- Integration test with Postgres fixture (read-only role)

### Phase 3 — Unified router + streaming
- Register `query_database` when active connection exists
- `stream_sql_agent_turn` wired into `/query/stream` (SSE `token` + `sql`)
- Sequential hybrid: SQL stream → RAG `stream_groq_answer`
- Router + tool tests

### Phase 4 — Chat UX polish
- SQL provenance panel; message persistence (`sql_queries`, `route_mode`)
- Tool status labels during SQL agent steps

### Phase 5 — Hardening
- Read-only role verification on connect (optional `SET default_transaction_read_only = on`)
- Connection pool eviction on disconnect
- Audit log table for executed SQL (user_id, query hash, timestamp)

---

## 10. Testing strategy

| Layer | Tests |
|-------|-------|
| API | connect/disconnect auth, invalid URL, test failure |
| SQL agent | mock LangChain agent; integration with fixture DB |
| Router | “revenue last quarter” → `query_database`; “chairwoman” → `search_documents`; disconnected → no SQL tool |
| Frontend | sidebar tick/cross; form validation; status after disconnect |
| Security | user A cannot read user B connection; logs contain no secrets |

---

## 11. Decisions (locked in)

| # | Decision | Resolution |
|---|----------|------------|
| 1 | Saved connections | **Multiple per user**; user picks which is **active**; only one active at a time |
| 2 | Answer voice | **Stream LangChain SQL agent output directly** via Groq — **no** repolish pass |
| 3 | Hybrid merge | **Sequential:** SQL first, then RAG, tokens in one assistant message |
| 4 | Dialect | **PostgreSQL only** in v1 |
| 5 | LangChain | **Current** `langchain` + `langchain-community` + `langchain-groq`; pin versions; `agent_type="tool-calling"` |
| 6 | SQL LLM | **Groq Cloud** through `ChatGroq` (`GROQ_API_KEY`, `SQL_AGENT_MODEL`) |
| 7 | Streaming | Reuse existing SSE **`token`** events for SQL and RAG text |

---

## 12. Files likely touched (reference)

| Area | Files |
|------|-------|
| Plan | `SQL_AGENT_PLAN.md` (this file) |
| Backend core | `app/sql_agent/*`, `app/api/sql_agent.py`, `app/db/models.py`, `app/main.py` |
| Agent integration | `app/llm/agent.py`, `app/llm/tools.py`, `app/api/query.py` |
| Config | `app/config.py`, `.env.example`, `requirements.txt` |
| Frontend | `App.tsx`, `api/sqlAgent.ts`, `components/SqlAgentPanel.tsx`, `components/SqlProvenancePanel.tsx`, `types.ts` |
| Tests | `tests/test_sql_agent*.py`, `tests/test_sql_connection_api.py` |

---

## 13. Success criteria

- [ ] Sidebar shows ✓/✗ based on **active** connection (saved-but-inactive = ✗).
- [ ] User can save **multiple** PostgreSQL connections and **activate** one at a time.
- [ ] No active connection → zero SQL behavior change in RAG.
- [ ] SQL answers stream via **`token` SSE** from LangChain + `ChatGroq` (Groq Cloud), no extra LLM pass.
- [ ] Hybrid questions run **SQL then RAG** sequentially in one streamed message.
- [ ] Router picks SQL vs RAG vs hybrid without manual mode toggle.
- [ ] Active connection’s **description** is injected into the SQL agent prompt.
- [ ] Credentials encrypted; PostgreSQL read-only user documented and tested.
