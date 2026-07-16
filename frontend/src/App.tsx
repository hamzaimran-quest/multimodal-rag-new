import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type ChatSessionSummary, createChat, deleteChat, getChat, listChats } from "./api/chats";
import { deleteDocument, isProcessing, uploadDocument } from "./api/client";
import { getSqlAgentStatus } from "./api/sqlAgent";
import { streamQuery, TOOL_LABELS } from "./api/query";
import { useAuth } from "./auth/AuthContext";
import { AssistantAvatar, UserAvatar } from "./components/Avatar";
import { ChatAssistantMessage } from "./components/ChatAssistantMessage";
import { EmbeddingModelBanner } from "./components/EmbeddingModelBanner";
import { IngestionProgressRing } from "./components/IngestionProgressRing";
import { PdfViewerBoundary } from "./components/PdfViewerBoundary";
import { PdfViewerPanel, type PdfViewerTarget } from "./components/PdfViewerPanel";
import { SpreadsheetViewerPanel, type SpreadsheetViewerTarget } from "./components/SpreadsheetViewerPanel";
import { SqlAgentPanel } from "./components/SqlAgentPanel";
import { useDocuments } from "./hooks/useDocuments";
import type { ComputedChart, DocumentRecord, QuerySource, SqlAgentStatus, SqlMeta } from "./types";
import { formatUploadDate } from "./utils/format";

const MAX_UPLOAD_FILES = 3;

type View = "chat" | "docs" | "sql";
interface Message {
  id?: number;
  role: "user" | "assistant";
  text: string;
  sources: QuerySource[];
  charts: ComputedChart[];
  sqlMeta?: SqlMeta | null;
  routeMode?: "sql" | "rag" | "hybrid" | null;
}


export default function App() {
  const { user, logout } = useAuth();
  const { documents, loading, error, refresh, setDocuments } = useDocuments();
  const [view, setView] = useState<View>("chat");
  const [menuOpen, setMenuOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [isNarrow, setIsNarrow] = useState(false);
  const [query, setQuery] = useState("");
  const [chatError, setChatError] = useState<string | null>(null);
  const [chatLoading, setChatLoading] = useState(false);
  const [agentToolStatus, setAgentToolStatus] = useState<string | null>(null);
  const [scopeDocIds, setScopeDocIds] = useState<string[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [chatSessions, setChatSessions] = useState<ChatSessionSummary[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<number | null>(null);
  const [chatsLoading, setChatsLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [docsError, setDocsError] = useState<string | null>(null);
  const [openSourcePanels, setOpenSourcePanels] = useState<Record<number, boolean>>({});
  const [openChartPanels, setOpenChartPanels] = useState<Record<number, boolean>>({});
  const [sqlAgentStatus, setSqlAgentStatus] = useState<SqlAgentStatus | null>(null);
  const [viewerTarget, setViewerTarget] = useState<PdfViewerTarget | null>(null);
  const [spreadsheetTarget, setSpreadsheetTarget] = useState<SpreadsheetViewerTarget | null>(null);
  const [headerMenuOpen, setHeaderMenuOpen] = useState(false);
  const [noSourceToast, setNoSourceToast] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const noSourceToastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const knownDocIds = useRef<Set<string>>(new Set());
  const chatScrollRef = useRef<HTMLDivElement | null>(null);
  const messageRefs = useRef<Record<number, HTMLDivElement | null>>({});
  const stickToBottomRef = useRef(true);
  const lastMessageCountRef = useRef(0);

  useEffect(() => {
    const mq = window.matchMedia("(max-width: 880px)");
    const sync = () => {
      setIsNarrow(mq.matches);
      if (!mq.matches) setMenuOpen(false);
    };
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, []);

  const closeMobileDrawer = useCallback(() => setMenuOpen(false), []);
  const openSidebarOrDrawer = useCallback(() => {
    if (isNarrow) setMenuOpen(true);
    else setSidebarOpen(true);
  }, [isNarrow]);
  const hideSidebarOrDrawer = useCallback(() => {
    if (isNarrow) setMenuOpen(false);
    else setSidebarOpen(false);
  }, [isNarrow]);

  useEffect(() => {
    if (!isNarrow || !menuOpen) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [isNarrow, menuOpen]);

  const openSourceInViewer = useCallback((messageSources: QuerySource[], source: QuerySource) => {
    if (!source.doc_id) {
      setView("docs");
      return;
    }
    if (source.source_format === "xlsx") {
      console.info("[spreadsheet-highlight] open_source", {
        chunkId: source.chunk_id,
        docId: source.doc_id,
        sheetName: source.sheet_name,
        rowRange: source.row_range,
        highlightRow: source.highlight_row,
        sheetRole: source.sheet_role,
      });
      setSpreadsheetTarget({
        docId: source.doc_id,
        filename: source.filename,
        sources: messageSources,
        chunkId: source.chunk_id,
        sheetName: source.sheet_name,
        sheetIndex: source.sheet_index,
        rowRange: source.row_range ?? null,
        highlightRow: source.highlight_row ?? null,
        sheetRole: source.sheet_role ?? null,
      });
      return;
    }
    const docRecord = documents.find((d) => d.doc_id === source.doc_id);
    const pageCount = source.page_count || docRecord?.page_count || 0;
    setViewerTarget({
      docId: source.doc_id,
      filename: source.filename,
      pageCount,
      sources: messageSources,
      chunkId: source.chunk_id,
      page:
        source.source_format === "docx"
          ? (source.viewer_page ?? 1)
          : Math.max(1, source.page_number || 1),
    });
  }, [documents]);

  const indexedDocs = useMemo(() => documents.filter((d) => d.ingestion_status === "indexed"), [documents]);
  const scopeAllSelected = indexedDocs.length === 0 || scopeDocIds.length === indexedDocs.length;
  const queryScopeDocIds = useMemo(() => {
    if (indexedDocs.length === 0) return undefined;
    if (scopeDocIds.length === 0 || scopeDocIds.length === indexedDocs.length) return undefined;
    return scopeDocIds;
  }, [indexedDocs.length, scopeDocIds]);
  const totalChunks = useMemo(() => documents.reduce((acc, doc) => acc + (doc.chunk_count || 0), 0), [documents]);
  const canSend = query.trim().length > 0 && !chatLoading;

  const refreshChatList = useCallback(async () => {
    try {
      const sessions = await listChats();
      setChatSessions(sessions);
    } catch {
      // keep existing list on transient failure
    }
  }, []);

  useEffect(() => {
    const ids = indexedDocs.map((d) => d.doc_id);
    setScopeDocIds((current) => {
      const kept = current.filter((id) => ids.includes(id));
      // Only auto-select docs that are new to the library — never re-add ids
      // the user has already seen and explicitly deselected (including down to zero).
      const added = ids.filter((id) => !knownDocIds.current.has(id));
      const next = [...kept, ...added];
      console.log("[scope-debug] sync-effect", {
        indexedDocIds: ids,
        currentScope: current,
        knownBefore: Array.from(knownDocIds.current),
        kept,
        added,
        nextScope: next,
      });
      knownDocIds.current = new Set(ids);
      return next;
    });
  }, [indexedDocs]);

  useEffect(() => {
    let active = true;
    getSqlAgentStatus()
      .then((status) => {
        if (active) setSqlAgentStatus(status);
      })
      .catch(() => {
        if (active) setSqlAgentStatus(null);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    setChatsLoading(true);
    listChats()
      .then((sessions) => {
        if (active) setChatSessions(sessions);
      })
      .catch(() => {
        if (active) setChatSessions([]);
      })
      .finally(() => {
        if (active) setChatsLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const handleNewChat = async () => {
    setChatError(null);
    try {
      const session = await createChat();
      setActiveSessionId(session.id);
      setMessages([]);
      setOpenSourcePanels({});
      setOpenChartPanels({});
      setView("chat");
      setMenuOpen(false);
      await refreshChatList();
    } catch (err) {
      setChatError(err instanceof Error ? err.message : "Failed to create chat");
    }
  };

  const handleSelectChat = async (sessionId: number) => {
    if (sessionId === activeSessionId && messages.length > 0) {
      setView("chat");
      setMenuOpen(false);
      return;
    }
    setChatError(null);
    try {
      const detail = await getChat(sessionId);
      setActiveSessionId(detail.id);
      setMessages(
        detail.messages.map((m) => ({
          id: m.id,
          role: m.role,
          text: m.content,
          sources: m.sources ?? [],
          charts: m.charts ?? [],
          sqlMeta: m.sql_meta ?? null,
          routeMode: (m.sql_meta?.route_mode as Message["routeMode"]) ?? (m.sources?.length ? "rag" : null),
        })),
      );
      setOpenSourcePanels({});
      setOpenChartPanels({});
      setView("chat");
      setMenuOpen(false);
    } catch (err) {
      setChatError(err instanceof Error ? err.message : "Failed to load chat");
    }
  };

  const handleDeleteChat = async (sessionId: number) => {
    try {
      await deleteChat(sessionId);
      if (activeSessionId === sessionId) {
        setActiveSessionId(null);
        setMessages([]);
        setOpenSourcePanels({});
        setOpenChartPanels({});
      }
      await refreshChatList();
    } catch (err) {
      setChatError(err instanceof Error ? err.message : "Failed to delete chat");
    }
  };

  useEffect(() => {
    return () => {
      if (noSourceToastTimer.current) clearTimeout(noSourceToastTimer.current);
    };
  }, []);

  const showNoSourceToast = useCallback(() => {
    setNoSourceToast(true);
    if (noSourceToastTimer.current) clearTimeout(noSourceToastTimer.current);
    noSourceToastTimer.current = setTimeout(() => setNoSourceToast(false), 5000);
  }, []);

  const handleComposerKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (canSend) void handleSend();
    }
  };

  // "Stick to bottom" tracking: while the assistant response streams in, keep
  // following the growing content -- unless the user scrolls up to read
  // something earlier, in which case auto-follow pauses until they scroll
  // back near the bottom themselves (the standard chat-app pattern).
  const handleChatScroll = useCallback(() => {
    const el = chatScrollRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    stickToBottomRef.current = distanceFromBottom < 120;
  }, []);

  // New message(s) appended: distinguish a live send (the user+assistant pair
  // handleSend appends) from a bulk history load (switching chats). For a
  // live send, scroll the newest user message to the top of the view -- like
  // ChatGPT/Claude -- so the incoming response has room to stream in below
  // it. For a bulk load (or an empty reset), jump straight to the bottom
  // instantly instead of animating through the whole history.
  useEffect(() => {
    const delta = messages.length - lastMessageCountRef.current;
    lastMessageCountRef.current = messages.length;
    if (delta <= 0) return;

    stickToBottomRef.current = true;
    if (delta <= 2) {
      for (let i = messages.length - 1; i >= 0; i -= 1) {
        if (messages[i].role === "user") {
          messageRefs.current[i]?.scrollIntoView({ behavior: "smooth", block: "start" });
          return;
        }
      }
    }
    const el = chatScrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages.length]);

  // Token/source/chart updates on the in-progress assistant message: follow
  // the bottom of the growing content, unless the user has scrolled away.
  useEffect(() => {
    if (!stickToBottomRef.current) return;
    const el = chatScrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages]);

  const handleSend = async () => {
    const prompt = query.trim();
    if (!prompt || chatLoading) return;
    const gateBlocked = !sqlAgentStatus?.has_active && scopeDocIds.length === 0;
    console.log("[scope-debug] handleSend gate", {
      scopeDocIds,
      indexedDocIds: indexedDocs.map((d) => d.doc_id),
      queryScopeDocIds,
      sqlHasActive: sqlAgentStatus?.has_active ?? null,
      gateBlocked,
    });
    if (gateBlocked) {
      showNoSourceToast();
      return;
    }
    setChatError(null);
    setChatLoading(true);
    setAgentToolStatus(null);
    setQuery("");
    setMessages((p) => [...p, { role: "user", text: prompt, sources: [], charts: [] }, { role: "assistant", text: "", sources: [], charts: [], sqlMeta: null, routeMode: null }]);
    // Buffer provenance until the answer stream finishes so route/SQL badges
    // do not appear before the reply text is complete.
    let pendingRoute: Message["routeMode"] = null;
    let pendingSql: SqlMeta | null = null;
    try {
      await streamQuery(
        {
          query: prompt,
          doc_ids: queryScopeDocIds,
          scope_empty: indexedDocs.length > 0 && scopeDocIds.length === 0,
          session_id: activeSessionId ?? undefined,
        },
        {
          onMeta: (meta) => {
            if (meta.session_id) setActiveSessionId(meta.session_id);
          },
          onRoute: (mode) => {
            pendingRoute = mode;
          },
          onTool: (tool) => {
            if (tool.status === "running") {
              setAgentToolStatus(tool.label ?? TOOL_LABELS[tool.name] ?? tool.name);
            }
          },
          onToken: (token) => {
            setAgentToolStatus(null);
            setMessages((prev) => {
              const next = [...prev];
              for (let i = next.length - 1; i >= 0; i -= 1) if (next[i].role === "assistant") { next[i] = { ...next[i], text: next[i].text + token }; break; }
              return next;
            });
          },
          onSources: (sources) => {
            const xlsxSources = sources.filter((source) => source.source_format === "xlsx");
            if (xlsxSources.length > 0) {
              console.info("[spreadsheet-highlight] sse_sources", xlsxSources.map((source) => ({
                chunkId: source.chunk_id,
                sheetName: source.sheet_name,
                sheetRole: source.sheet_role,
                rowRange: source.row_range,
                highlightRow: source.highlight_row,
              })));
            }
            setMessages((prev) => {
              const next = [...prev];
              for (let i = next.length - 1; i >= 0; i -= 1) {
                if (next[i].role === "assistant") {
                  next[i] = { ...next[i], sources };
                  break;
                }
              }
              return next;
            });
          },
          onCharts: (charts) => setMessages((prev) => {
            const next = [...prev];
            for (let i = next.length - 1; i >= 0; i -= 1) if (next[i].role === "assistant") { next[i] = { ...next[i], charts }; break; }
            return next;
          }),
          onSql: (sqlMeta) => {
            pendingSql = sqlMeta;
            if (sqlMeta.route_mode) {
              pendingRoute = sqlMeta.route_mode as Message["routeMode"];
            }
          },
          onError: (message) => setChatError(message),
        },
      );
      if (pendingRoute || pendingSql) {
        setMessages((prev) => {
          const next = [...prev];
          for (let i = next.length - 1; i >= 0; i -= 1) {
            if (next[i].role === "assistant") {
              next[i] = {
                ...next[i],
                ...(pendingRoute ? { routeMode: pendingRoute } : {}),
                ...(pendingSql ? { sqlMeta: pendingSql } : {}),
              };
              break;
            }
          }
          return next;
        });
      }
      await refreshChatList();
    } catch (err) {
      setChatError(err instanceof Error ? err.message : "Streaming failed");
    } finally {
      setChatLoading(false);
      setAgentToolStatus(null);
    }
  };

  const uploadOne = async (file: File) => {
    if (!/\.(pdf|docx|xlsx)$/i.test(file.name)) {
      setDocsError(`${file.name}: only PDF, DOCX, and XLSX files are supported.`);
      return;
    }
    try {
      const result = await uploadDocument(file);
      const optimistic: DocumentRecord = {
        doc_id: result.doc_id, filename: result.filename, ingestion_status: "processing", ingestion_progress: 2,
        progress_message: "Queued", upload_timestamp: new Date().toISOString(), chunk_count: 0, error_message: null,
      };
      setDocuments((current) => [optimistic, ...current.filter((d) => d.doc_id !== optimistic.doc_id)]);
    } catch (err) {
      setDocsError(`${file.name}: ${err instanceof Error ? err.message : "Upload failed"}`);
    }
  };

  const handleUpload = async (fileList?: FileList | null) => {
    if (!fileList || fileList.length === 0) return;
    const files = Array.from(fileList).slice(0, MAX_UPLOAD_FILES);
    setDocsError(
      fileList.length > MAX_UPLOAD_FILES
        ? `You can upload up to ${MAX_UPLOAD_FILES} files at a time; only the first ${MAX_UPLOAD_FILES} were queued.`
        : null,
    );
    setUploading(true);
    try {
      await Promise.all(files.map(uploadOne));
      await refresh();
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleDeleteDocument = async (doc: DocumentRecord) => {
    setDocsError(null);
    try {
      await deleteDocument(doc.doc_id);
      setDocuments((current) => current.filter((d) => d.doc_id !== doc.doc_id));
      setScopeDocIds((current) => current.filter((id) => id !== doc.doc_id));
    } catch (err) {
      setDocsError(err instanceof Error ? err.message : "Delete failed");
    }
  };

  return (
    <div className="h-[100dvh] h-screen overflow-hidden bg-[radial-gradient(ellipse_80%_50%_at_50%_-20%,rgba(120,120,120,0.15),transparent),radial-gradient(ellipse_60%_40%_at_100%_100%,rgba(80,80,80,0.08),transparent),#0a0a0a] text-[#e5e5e5]">
      <EmbeddingModelBanner />
      <div className={`grid h-full transition-all duration-200 ${sidebarOpen ? "grid-cols-[272px_1fr]" : "grid-cols-[0_1fr]"} max-[880px]:grid-cols-1`}>
        {isNarrow && menuOpen && (
          <button
            type="button"
            aria-label="Close menu"
            className="fixed inset-0 z-30 cursor-default bg-black/50"
            onClick={closeMobileDrawer}
          />
        )}
        <aside
          aria-hidden={isNarrow ? !menuOpen : !sidebarOpen}
          className={[
            "overflow-y-auto border-r border-[#2a2a2a] bg-gradient-to-b from-[#111111] to-[#0d0d0d] transition-all duration-200",
            sidebarOpen ? "px-4 pb-4 pt-5 opacity-100" : "w-0 overflow-hidden px-0 py-0 opacity-0",
            // Mobile drawer: always full drawer chrome; visibility via translate only.
            "max-[880px]:fixed max-[880px]:inset-y-0 max-[880px]:left-0 max-[880px]:z-40",
            "max-[880px]:w-[min(288px,82vw)] max-[880px]:overflow-y-auto max-[880px]:px-4 max-[880px]:pb-4 max-[880px]:pt-5 max-[880px]:opacity-100",
            "max-[880px]:transition-transform max-[880px]:duration-200",
            menuOpen ? "max-[880px]:translate-x-0 max-[880px]:shadow-2xl max-[880px]:shadow-black/50" : "max-[880px]:-translate-x-full",
          ].join(" ")}
        >
          <div className="mb-5 flex items-center justify-between">
            <div className="flex h-[34px] w-[34px] items-center justify-center rounded-[10px] bg-gradient-to-br from-[#525252] via-[#404040] to-[#262626] font-['Space_Grotesk'] text-base font-bold text-[#e5e5e5] shadow-lg shadow-black/40">M</div>
            <button
              type="button"
              onClick={hideSidebarOrDrawer}
              className="flex h-[34px] w-[34px] items-center justify-center rounded-[8px] text-[#a3a3a3] hover:bg-[#1f1f1f] hover:text-[#e5e5e5]"
              aria-label={isNarrow ? "Close menu" : "Hide sidebar"}
              title={isNarrow ? "Close menu" : "Hide sidebar"}
            >
              <svg width="19" height="19" viewBox="0 0 24 24" fill="none" aria-hidden>
                <rect x="3" y="4" width="18" height="16" rx="2.5" stroke="currentColor" strokeWidth="1.7" />
                <line x1="9" y1="4" x2="9" y2="20" stroke="currentColor" strokeWidth="1.7" />
              </svg>
            </button>
          </div>
          <div className="mb-5 space-y-0.5">
            <button
              type="button"
              onClick={() => void handleNewChat()}
              className="flex w-full items-center gap-2.5 rounded-[8px] px-2.5 py-2.5 text-left text-[14px] font-medium text-[#e5e5e5] hover:bg-[#1f1f1f]"
              data-testid="new-chat-button"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden className="shrink-0 text-[#a3a3a3]">
                <path d="M4 20h4L18.5 9.5a2 2 0 0 0-2.83-2.83L5 17.17 4 20z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
                <path d="M13.5 6.5l4 4" stroke="currentColor" strokeWidth="1.6" />
              </svg>
              New chat
            </button>
            <button
              type="button"
              onClick={() => { setView("chat"); setMenuOpen(false); }}
              className={`flex w-full items-center gap-2.5 rounded-[8px] px-2.5 py-2.5 text-left text-[14px] font-medium ${view === "chat" ? "bg-[#262626] text-[#f5f5f5]" : "text-[#a3a3a3] hover:bg-[#1f1f1f] hover:text-[#e5e5e5]"}`}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden className="shrink-0">
                <path d="M4 5h16v11H8l-4 3V5z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
              </svg>
              Chat
            </button>
            <button
              type="button"
              onClick={() => { setView("docs"); setMenuOpen(false); }}
              className={`flex w-full items-center gap-2.5 rounded-[8px] px-2.5 py-2.5 text-left text-[14px] font-medium ${view === "docs" ? "bg-[#262626] text-[#f5f5f5]" : "text-[#a3a3a3] hover:bg-[#1f1f1f] hover:text-[#e5e5e5]"}`}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden className="shrink-0">
                <path d="M3 6a1 1 0 0 1 1-1h5l2 2h8a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
              </svg>
              Documents
              <span className="ml-auto rounded-full bg-[#262626] px-1.5 text-[11px] text-[#737373]">{documents.length}</span>
            </button>
            <button
              type="button"
              onClick={() => { setView("sql"); setMenuOpen(false); }}
              className={`flex w-full items-center gap-2.5 rounded-[8px] px-2.5 py-2.5 text-left text-[14px] font-medium ${view === "sql" ? "bg-[#262626] text-[#f5f5f5]" : "text-[#a3a3a3] hover:bg-[#1f1f1f] hover:text-[#e5e5e5]"}`}
              data-testid="sql-agent-nav"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden className="shrink-0">
                <ellipse cx="12" cy="6" rx="7" ry="3" stroke="currentColor" strokeWidth="1.6" />
                <path d="M5 6v12c0 1.66 3.13 3 7 3s7-1.34 7-3V6" stroke="currentColor" strokeWidth="1.6" />
                <path d="M5 12c0 1.66 3.13 3 7 3s7-1.34 7-3" stroke="currentColor" strokeWidth="1.6" />
              </svg>
              SQL Agent
              <span
                className={`ml-auto flex h-6 w-6 shrink-0 items-center justify-center rounded-full ${
                  sqlAgentStatus?.has_active
                    ? "bg-emerald-500/20 text-emerald-400"
                    : "bg-rose-500/20 text-rose-400"
                }`}
                aria-label={sqlAgentStatus?.has_active ? "Database connected" : "No active database"}
                title={sqlAgentStatus?.has_active ? "Active connection" : "No active connection"}
              >
                {sqlAgentStatus?.has_active ? (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden>
                    <path d="M5 13l4 4L19 7" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                ) : (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden>
                    <path d="M7 7l10 10M17 7L7 17" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
                  </svg>
                )}
              </span>
            </button>
          </div>
          <div className="mt-4 border-t border-[#2a2a2a] pt-4">
            <p className="mb-2 px-1 text-[15px] font-bold text-[#f5f5f5]">Conversations</p>
            {chatsLoading ? (
              <p className="px-2 text-[12px] text-[#737373]">Loading...</p>
            ) : chatSessions.length === 0 ? (
              <p className="px-2 text-[12px] text-[#737373]">No conversations yet</p>
            ) : (
              <div className="max-h-[220px] space-y-0.5 overflow-y-auto">
                {chatSessions.map((session) => (
                  <div
                    key={session.id}
                    className={`group flex items-center gap-1 rounded-[8px] px-1 ${activeSessionId === session.id ? "bg-[#262626]" : "hover:bg-[#1a1a1a]"}`}
                  >
                    <button
                      type="button"
                      onClick={() => void handleSelectChat(session.id)}
                      className="min-w-0 flex-1 truncate rounded-[8px] px-2 py-2 text-left text-[12.5px] text-[#d4d4d4] hover:text-[#f5f5f5]"
                      data-testid={`chat-session-${session.id}`}
                      title={session.title}
                    >
                      {session.title}
                    </button>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        void handleDeleteChat(session.id);
                      }}
                      className="rounded-[6px] px-1.5 py-1 text-[11px] text-[#525252] opacity-0 transition-opacity hover:text-rose-300 group-hover:opacity-100 max-[880px]:opacity-100"
                      aria-label={`Delete ${session.title}`}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
          <div className="mt-5 border-t border-[#2a2a2a] pt-4">
            <p className="mb-2 px-1 text-[15px] font-bold text-[#f5f5f5]">Retrieval</p>
            <label className="mb-1.5 block px-1 text-[11.5px] font-medium text-[#a3a3a3]">Scope</label>
            <div className="max-h-40 space-y-1 overflow-y-auto rounded-[8px] border border-[#333333] bg-[#1a1a1a] px-2 py-2">
              <label className="flex cursor-pointer items-center gap-2 rounded-[6px] px-1.5 py-1 hover:bg-[#242424]">
                <input
                  type="checkbox"
                  checked={scopeAllSelected}
                  onChange={() => {
                    const next = scopeAllSelected
                      ? (indexedDocs[0] ? [indexedDocs[0].doc_id] : [])
                      : indexedDocs.map((d) => d.doc_id);
                    console.log("[scope-debug] toggle all-documents checkbox", { wasAllSelected: scopeAllSelected, next });
                    setScopeDocIds(next);
                  }}
                  className="accent-[#d4d4d4]"
                />
                <span className="text-[12.5px] font-medium text-[#e5e5e5]">All documents</span>
              </label>
              {indexedDocs.map((doc) => (
                <label key={doc.doc_id} className="flex cursor-pointer items-center gap-2 rounded-[6px] px-1.5 py-1 hover:bg-[#242424]">
                  <input
                    type="checkbox"
                    checked={scopeDocIds.includes(doc.doc_id)}
                    onChange={() => {
                      setScopeDocIds((current) => {
                        const next = current.includes(doc.doc_id)
                          ? current.filter((id) => id !== doc.doc_id)
                          : [...current, doc.doc_id];
                        console.log("[scope-debug] toggle single doc checkbox", { docId: doc.doc_id, current, next });
                        return next;
                      });
                    }}
                    className="accent-[#d4d4d4]"
                  />
                  <span className="truncate text-[12px] text-[#d4d4d4]" title={doc.filename}>{doc.filename}</span>
                </label>
              ))}
              {indexedDocs.length === 0 && (
                <p className="px-1.5 py-1 text-[11.5px] text-[#737373]">No indexed documents yet.</p>
              )}
            </div>
          </div>
        </aside>

        <main className="flex min-w-0 flex-col overflow-hidden bg-gradient-to-b from-[#0f0f0f] to-[#0a0a0a]">
          <div className="shrink-0 border-b border-[#2a2a2a] px-4 py-2.5 max-[880px]:px-3 max-[880px]:py-2">
            <div className="flex items-center justify-between gap-2">
              <div className="flex min-w-0 items-center gap-1">
              {((isNarrow && !menuOpen) || (!isNarrow && !sidebarOpen)) && (
                <button
                  type="button"
                  onClick={openSidebarOrDrawer}
                  className="flex h-[40px] w-[40px] shrink-0 items-center justify-center rounded-[8px] text-[#a3a3a3] hover:bg-[#1f1f1f] hover:text-[#e5e5e5] max-[880px]:h-11 max-[880px]:w-11"
                  aria-label={isNarrow ? "Open menu" : "Show sidebar"}
                  title={isNarrow ? "Open menu" : "Show sidebar"}
                  data-testid="open-sidebar-button"
                >
                  <svg width="19" height="19" viewBox="0 0 24 24" fill="none" aria-hidden>
                    <rect x="3" y="4" width="18" height="16" rx="2.5" stroke="currentColor" strokeWidth="1.7" />
                    <line x1="9" y1="4" x2="9" y2="20" stroke="currentColor" strokeWidth="1.7" />
                  </svg>
                </button>
              )}

              <div className="relative min-w-0">
                <button
                  type="button"
                  onClick={() => setHeaderMenuOpen((v) => !v)}
                  className="flex max-w-full items-center gap-1.5 truncate rounded-[8px] px-2.5 py-1.5 text-[16px] font-semibold text-[#f5f5f5] hover:bg-[#1f1f1f] max-[880px]:text-[15px]"
                  aria-haspopup="menu"
                  aria-expanded={headerMenuOpen}
                >
                  <span className="truncate">Multimodal RAG</span>
                  <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden className={`shrink-0 text-[#a3a3a3] transition-transform ${headerMenuOpen ? "rotate-180" : ""}`}>
                    <path d="M4 6l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>

                {headerMenuOpen && (
                  <>
                    <button type="button" aria-hidden className="fixed inset-0 z-30 cursor-default" onClick={() => setHeaderMenuOpen(false)} />
                    <div className="absolute left-0 z-40 mt-1.5 w-[248px] rounded-[12px] border border-[#2a2a2a] bg-[#161616] p-1.5 shadow-2xl shadow-black/50" role="menu">
                      <div className="px-2.5 py-1.5 text-[11px] font-medium text-[#737373]">Document Q&amp;A · grounded, cited answers</div>
                      <div className="my-1 h-px bg-[#2a2a2a]" />
                      <div className="flex items-center justify-between rounded-[8px] px-2.5 py-2 hover:bg-[#1f1f1f]">
                        <span className="text-[13px] text-[#d4d4d4]">Documents</span>
                        <span className="font-['Space_Grotesk'] text-[13px] font-semibold text-[#f5f5f5]">{documents.length}</span>
                      </div>
                      <div className="flex items-center justify-between rounded-[8px] px-2.5 py-2 hover:bg-[#1f1f1f]">
                        <span className="text-[13px] text-[#d4d4d4]">Indexed chunks</span>
                        <span className="font-['Space_Grotesk'] text-[13px] font-semibold text-[#f5f5f5]">{totalChunks}</span>
                      </div>
                    </div>
                  </>
                )}
              </div>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                <span className="hidden max-w-[160px] truncate text-[12.5px] text-[#a3a3a3] sm:inline" title={user?.email}>{user?.email}</span>
                <button
                  type="button"
                  onClick={() => void logout()}
                  className="flex items-center gap-1.5 rounded-[8px] border border-[#333333] px-2.5 py-1.5 text-[12px] font-medium text-[#a3a3a3] hover:border-[#525252] hover:text-[#e5e5e5] max-[880px]:h-11 max-[880px]:w-11 max-[880px]:justify-center max-[880px]:px-0"
                  data-testid="logout-button"
                  aria-label="Log out"
                  title="Log out"
                >
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden>
                    <path d="M15 4h3a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1h-3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                    <path d="M10 8l-4 4 4 4M6 12h10" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <span className="max-[880px]:hidden">Log out</span>
                </button>
              </div>
            </div>
          </div>

          {view === "chat" ? (
            <section className="flex min-h-0 flex-1 flex-col">
              <div
                ref={chatScrollRef}
                onScroll={handleChatScroll}
                className="flex-1 space-y-6 overflow-y-auto px-7 pb-2 pt-6 max-[880px]:space-y-4 max-[880px]:px-3 max-[880px]:pt-4"
              >
                {messages.map((msg, idx) => (
                  <div
                    key={msg.id ?? `${msg.role}-${idx}`}
                    ref={(el) => {
                      messageRefs.current[idx] = el;
                    }}
                    className={`flex items-start gap-2.5 max-[880px]:gap-2 ${msg.role === "user" ? "justify-end" : ""}`}
                  >
                    {msg.role === "assistant" && <AssistantAvatar className="mt-0.5" />}
                    {msg.role === "user" ? (
                      <div className="max-w-[66%] rounded-[16px_16px_4px_16px] bg-gradient-to-br from-[#404040] to-[#2a2a2a] px-4 py-3 text-[14px] leading-relaxed text-[#f5f5f5] shadow-lg shadow-black/20 max-[880px]:max-w-[80%] max-[880px]:px-3.5 max-[880px]:py-2.5 max-[880px]:text-[13.5px] max-[880px]:leading-[1.55]">{msg.text}</div>
                    ) : (
                      <ChatAssistantMessage
                        messageIndex={idx}
                        text={msg.text}
                        sources={msg.sources}
                        charts={msg.charts}
                        sqlMeta={msg.sqlMeta}
                        routeMode={msg.routeMode}
                        placeholder={
                          chatLoading && idx === messages.length - 1
                            ? (agentToolStatus ?? "...")
                            : ""
                        }
                        sourcesOpen={!!openSourcePanels[idx]}
                        chartsOpen={!!openChartPanels[idx]}
                        onToggleSources={() => setOpenSourcePanels((prev) => ({ ...prev, [idx]: !prev[idx] }))}
                        onToggleCharts={() => setOpenChartPanels((prev) => ({ ...prev, [idx]: !prev[idx] }))}
                        onGoToPage={() => setView("docs")}
                        onOpenSource={(source) => openSourceInViewer(msg.sources, source)}
                      />
                    )}
                    {msg.role === "user" && <UserAvatar className="mt-0.5" />}
                  </div>
                ))}
              </div>
              <div className="shrink-0 border-t border-[#2a2a2a] bg-gradient-to-t from-[#0a0a0a] to-transparent px-7 pb-[max(1.25rem,env(safe-area-inset-bottom))] pt-4 max-[880px]:px-3 max-[880px]:pt-3">
                {chatError && <div className="mb-2 rounded border border-rose-500/35 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">{chatError}</div>}
                <div className="mb-2 hidden text-[12px] text-[#737373] min-[881px]:block">try — "Compare Timberland Bancorp&apos;s net interest margin across the last two fiscal years" · <span className="text-[#525252]">Enter to send, Shift+Enter for new line</span></div>
                <div className="mb-2 text-[11.5px] text-[#737373] min-[881px]:hidden">Enter to send · Shift+Enter for new line</div>
                <div className="flex items-end gap-2.5 max-[880px]:gap-2">
                  <textarea
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={handleComposerKeyDown}
                    placeholder="Ask about your documents…"
                    rows={1}
                    className="min-h-[48px] max-h-[140px] flex-1 resize-none rounded-[14px] border border-[#333333] bg-gradient-to-b from-[#1a1a1a] to-[#141414] px-4 py-3 text-[14px] text-[#e5e5e5] outline-none placeholder:text-[#525252] focus:border-[#525252] focus:ring-1 focus:ring-[#404040] max-[880px]:min-h-[52px] max-[880px]:text-[16px]"
                    data-testid="chat-input"
                  />
                  <button
                    type="button"
                    onClick={() => void handleSend()}
                    disabled={!canSend}
                    className="h-[48px] shrink-0 rounded-[14px] bg-gradient-to-b from-[#525252] to-[#333333] px-6 text-[13px] font-semibold text-[#f5f5f5] shadow-lg shadow-black/30 hover:from-[#737373] hover:to-[#404040] disabled:opacity-40 disabled:hover:from-[#525252] disabled:hover:to-[#333333] max-[880px]:h-[52px] max-[880px]:min-w-[52px] max-[880px]:px-4"
                    data-testid="chat-send"
                    aria-label={chatLoading ? "Streaming" : "Send"}
                  >
                    <span className="max-[880px]:hidden">{chatLoading ? "Streaming..." : "Send"}</span>
                    <span className="hidden max-[880px]:inline">{chatLoading ? "…" : "↑"}</span>
                  </button>
                </div>
              </div>
            </section>
          ) : view === "docs" ? (
            <section className="flex-1 overflow-y-auto px-7 pb-7 pt-6 max-[880px]:px-3 max-[880px]:pb-6 max-[880px]:pt-4">
              <div className="mb-5 flex items-start justify-between gap-3 max-[880px]:mb-4">
                <div className="min-w-0">
                  <h2 className="font-['Space_Grotesk'] text-[20px] font-semibold text-[#f5f5f5] max-[880px]:text-[18px]">Documents</h2>
                  <p className="mt-1 text-[13px] text-[#a3a3a3] max-[880px]:hidden">Upload PDF, DOCX, or XLSX files for ingestion. Status updates automatically while processing.</p>
                </div>
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={uploading}
                  className="hidden shrink-0 rounded-[8px] bg-gradient-to-b from-[#525252] to-[#333333] px-3.5 py-2 text-[12.5px] font-semibold text-[#f5f5f5] shadow-lg shadow-black/30 hover:from-[#737373] hover:to-[#404040] disabled:opacity-40 max-[880px]:inline-flex max-[880px]:min-h-10 max-[880px]:items-center"
                >
                  {uploading ? "Uploading…" : "Upload"}
                </button>
              </div>
              <input ref={fileInputRef} type="file" multiple accept="application/pdf,.pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,.docx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,.xlsx" className="hidden" onChange={(e) => void handleUpload(e.target.files)} />
              <div
                className="mb-6 rounded-[20px] border border-dashed border-[#333333] bg-gradient-to-b from-[#141414] to-[#0f0f0f] px-5 py-10 text-center hover:border-[#525252] transition-colors max-[880px]:mb-3 max-[880px]:hidden"
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => { e.preventDefault(); void handleUpload(e.dataTransfer.files); }}
              >
                <p className="mb-4 text-[13.5px] text-[#a3a3a3]">Drag up to {MAX_UPLOAD_FILES} PDF, DOCX, or XLSX files here, or choose files to index text and tables for retrieval.</p>
                <button type="button" onClick={() => fileInputRef.current?.click()} className="rounded-[8px] bg-gradient-to-b from-[#525252] to-[#333333] px-5 py-2.5 text-[13px] font-semibold text-[#f5f5f5] shadow-lg shadow-black/30 hover:from-[#737373] hover:to-[#404040]">{uploading ? "Uploading..." : "Choose files"}</button>
              </div>
              {(docsError || error) && <div className="mb-3 rounded border border-rose-500/35 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">{docsError || error}</div>}
              {loading ? <p className="text-sm text-[#a3a3a3]">Loading documents...</p> : documents.length === 0 ? (
                <p className="text-sm text-[#737373] max-[880px]:pt-2">No documents yet. Tap Upload to add a file.</p>
              ) : (
                <div className="space-y-2.5 max-[880px]:space-y-2">
                  {documents.map((doc) => (
                    <div key={doc.doc_id} data-testid={`doc-row-${doc.doc_id}`} className="flex items-center gap-4 rounded-[14px] border border-[#2a2a2a] bg-gradient-to-b from-[#1a1a1a] to-[#141414] px-4 py-3 hover:border-[#404040] transition-colors max-[880px]:gap-2.5 max-[880px]:px-3 max-[880px]:py-2.5">
                      <div className="flex min-w-0 flex-1 items-center gap-3 max-[880px]:gap-2.5">
                        <IngestionProgressRing
                          status={doc.ingestion_status}
                          uploadTimestamp={doc.upload_timestamp}
                          backendProgress={doc.ingestion_progress}
                          docId={doc.doc_id}
                        />
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-[13.5px] font-semibold text-[#e5e5e5] max-[880px]:text-[13px]">{doc.filename}</p>
                          <p className="truncate text-[11.5px] text-[#737373] max-[880px]:text-[11px]">
                            <span className="max-[880px]:hidden">{formatUploadDate(doc.upload_timestamp)}</span>
                            <span className="hidden max-[880px]:inline">
                              {doc.ingestion_status}
                              {doc.chunk_count ? ` · ${doc.chunk_count} chunks` : ""}
                            </span>
                          </p>
                          {doc.progress_message && doc.ingestion_status !== "failed" && (
                            <p className="mt-0.5 truncate text-[11px] text-[#a3a3a3] max-[880px]:hidden">{doc.progress_message}</p>
                          )}
                          {doc.error_message && (
                            <p className="mt-0.5 line-clamp-2 text-[11px] text-rose-300">{doc.error_message}</p>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-3 max-[880px]:gap-1">
                        <span className={`rounded-full px-2.5 py-1 text-[11px] font-semibold max-[880px]:hidden ${doc.ingestion_status === "indexed" ? "bg-[#404040]/40 text-[#d4d4d4]" : isProcessing(doc.ingestion_status) ? "bg-[#525252]/30 text-[#a3a3a3]" : "bg-rose-500/15 text-rose-300"}`}>{doc.ingestion_status}</span>
                        <span className="min-w-10 text-right font-['Space_Grotesk'] text-[13px] font-semibold text-[#d4d4d4] max-[880px]:hidden">{doc.chunk_count || "—"}</span>
                        <button
                          type="button"
                          onClick={() => void handleDeleteDocument(doc)}
                          className="min-h-9 px-1 text-xs font-medium text-[#a3a3a3] hover:text-rose-300 hover:underline max-[880px]:min-h-10 max-[880px]:min-w-10 max-[880px]:px-2"
                          aria-label={`Delete ${doc.filename}`}
                        >
                          <span className="max-[880px]:hidden">Delete</span>
                          <span className="hidden text-[18px] leading-none text-[#737373] max-[880px]:inline" aria-hidden>×</span>
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </section>
          ) : (
            <SqlAgentPanel onStatusChange={setSqlAgentStatus} />
          )}
        </main>
      </div>
      {viewerTarget && (
        <PdfViewerBoundary
          onClose={() => setViewerTarget(null)}
          resetKey={`${viewerTarget.docId}:${viewerTarget.chunkId}`}
        >
          <PdfViewerPanel target={viewerTarget} onClose={() => setViewerTarget(null)} />
        </PdfViewerBoundary>
      )}
      {spreadsheetTarget && (
        <SpreadsheetViewerPanel
          target={spreadsheetTarget}
          onClose={() => setSpreadsheetTarget(null)}
        />
      )}
      {noSourceToast && (
        <div
          role="status"
          className="auth-fade-in fixed bottom-5 right-5 z-50 max-w-[320px] rounded-[12px] border border-[#2a2a2a] bg-gradient-to-br from-[#1a1a1a] to-[#111111] px-4 py-3.5 text-[13.5px] leading-relaxed text-[#e5e5e5] shadow-2xl shadow-black/60"
        >
          <p className="font-medium text-[#f5f5f5]">Nothing to search yet</p>
          <p className="mt-1 text-[#a3a3a3]">Please set up a database connection or upload documents to get started.</p>
        </div>
      )}
    </div>
  );
}
