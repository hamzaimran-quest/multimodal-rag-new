import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type ChatSessionSummary, createChat, deleteChat, getChat, listChats } from "./api/chats";
import { deleteDocument, isProcessing, uploadDocument } from "./api/client";
import { streamQuery } from "./api/query";
import { useAuth } from "./auth/AuthContext";
import { ComputedChartsPanel } from "./components/ComputedChartsPanel";
import { HeroImages } from "./components/HeroImages";
import { IngestionProgressRing } from "./components/IngestionProgressRing";
import { MarkdownAnswer } from "./components/MarkdownAnswer";
import { PdfViewerBoundary } from "./components/PdfViewerBoundary";
import { PdfViewerPanel, type PdfViewerTarget } from "./components/PdfViewerPanel";
import { SourcesPanel } from "./components/SourcesPanel";
import { useDocuments } from "./hooks/useDocuments";
import { deriveHeroImages } from "./lib/heroImages";
import type { ComputedChart, DocumentRecord, QuerySource } from "./types";
import { formatUploadDate } from "./utils/format";

type View = "chat" | "docs";
interface Message {
  id?: number;
  role: "user" | "assistant";
  text: string;
  sources: QuerySource[];
  charts: ComputedChart[];
}

export default function App() {
  const { user, logout } = useAuth();
  const { documents, loading, error, refresh, setDocuments } = useDocuments();
  const [view, setView] = useState<View>("chat");
  const [menuOpen, setMenuOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [query, setQuery] = useState("");
  const [chatError, setChatError] = useState<string | null>(null);
  const [chatLoading, setChatLoading] = useState(false);
  const [scopeDocId, setScopeDocId] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [chatSessions, setChatSessions] = useState<ChatSessionSummary[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<number | null>(null);
  const [chatsLoading, setChatsLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [docsError, setDocsError] = useState<string | null>(null);
  const [openSourcePanels, setOpenSourcePanels] = useState<Record<number, boolean>>({});
  const [openChartPanels, setOpenChartPanels] = useState<Record<number, boolean>>({});
  const [viewerTarget, setViewerTarget] = useState<PdfViewerTarget | null>(null);
  const [headerMenuOpen, setHeaderMenuOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const openSourceInViewer = useCallback((messageSources: QuerySource[], source: QuerySource) => {
    if (!source.doc_id) {
      setView("docs");
      return;
    }
    setViewerTarget({
      docId: source.doc_id,
      filename: source.filename,
      pageCount: source.page_count ?? 0,
      sources: messageSources,
      chunkId: source.chunk_id,
      page:
        source.source_format === "docx"
          ? (source.viewer_page ?? 1)
          : source.page_number,
    });
  }, []);

  const indexedDocs = useMemo(() => documents.filter((d) => d.ingestion_status === "indexed"), [documents]);
  const latestLibraryDocs = useMemo(() => documents.slice(0, 3), [documents]);
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

  const handleComposerKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (canSend) void handleSend();
    }
  };

  const handleSend = async () => {
    const prompt = query.trim();
    if (!prompt || chatLoading) return;
    setChatError(null);
    setChatLoading(true);
    setQuery("");
    setMessages((p) => [...p, { role: "user", text: prompt, sources: [], charts: [] }, { role: "assistant", text: "", sources: [], charts: [] }]);
    try {
      await streamQuery(
        { query: prompt, doc_id: scopeDocId || undefined, session_id: activeSessionId ?? undefined },
        {
          onMeta: (meta) => {
            if (meta.session_id) setActiveSessionId(meta.session_id);
          },
          onToken: (token) => setMessages((prev) => {
            const next = [...prev];
            for (let i = next.length - 1; i >= 0; i -= 1) if (next[i].role === "assistant") { next[i] = { ...next[i], text: next[i].text + token }; break; }
            return next;
          }),
          onSources: (sources) => setMessages((prev) => {
            const next = [...prev];
            for (let i = next.length - 1; i >= 0; i -= 1) if (next[i].role === "assistant") { next[i] = { ...next[i], sources }; break; }
            return next;
          }),
          onCharts: (charts) => setMessages((prev) => {
            const next = [...prev];
            for (let i = next.length - 1; i >= 0; i -= 1) if (next[i].role === "assistant") { next[i] = { ...next[i], charts }; break; }
            return next;
          }),
          onError: (message) => setChatError(message),
        },
      );
      await refreshChatList();
    } catch (err) {
      setChatError(err instanceof Error ? err.message : "Streaming failed");
    } finally {
      setChatLoading(false);
    }
  };

  const handleUpload = async (file?: File) => {
    if (!file) return;
    if (!/\.(pdf|docx)$/i.test(file.name)) return setDocsError("Only PDF and DOCX files are supported.");
    setDocsError(null);
    setUploading(true);
    try {
      const result = await uploadDocument(file);
      const optimistic: DocumentRecord = {
        doc_id: result.doc_id, filename: result.filename, ingestion_status: "processing", ingestion_progress: 2,
        progress_message: "Queued", upload_timestamp: new Date().toISOString(), chunk_count: 0, error_message: null,
      };
      setDocuments((current) => [optimistic, ...current.filter((d) => d.doc_id !== optimistic.doc_id)]);
      await refresh();
    } catch (err) {
      setDocsError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  return (
    <div className="h-screen overflow-hidden bg-[radial-gradient(ellipse_80%_50%_at_50%_-20%,rgba(120,120,120,0.15),transparent),radial-gradient(ellipse_60%_40%_at_100%_100%,rgba(80,80,80,0.08),transparent),#0a0a0a] text-[#e5e5e5]">
      <div className={`grid h-full transition-all duration-200 ${sidebarOpen ? "grid-cols-[272px_1fr]" : "grid-cols-[0_1fr]"} max-[880px]:grid-cols-1`}>
        <aside className={`overflow-y-auto border-r border-[#2a2a2a] bg-gradient-to-b from-[#111111] to-[#0d0d0d] transition-all duration-200 ${sidebarOpen ? "px-4 pb-4 pt-5 opacity-100" : "w-0 px-0 py-0 opacity-0"} max-[880px]:fixed max-[880px]:inset-y-0 max-[880px]:left-0 max-[880px]:z-20 max-[880px]:w-[70%] max-[880px]:-translate-x-full max-[880px]:transition ${menuOpen ? "max-[880px]:translate-x-0" : ""}`}>
          <div className="mb-5 flex items-center justify-between">
            <div className="flex h-[34px] w-[34px] items-center justify-center rounded-[10px] bg-gradient-to-br from-[#525252] via-[#404040] to-[#262626] font-['Space_Grotesk'] text-base font-bold text-[#e5e5e5] shadow-lg shadow-black/40">M</div>
            <button
              type="button"
              onClick={() => { setSidebarOpen((v) => !v); setMenuOpen((v) => !v); }}
              className="flex h-[34px] w-[34px] items-center justify-center rounded-[8px] text-[#a3a3a3] hover:bg-[#1f1f1f] hover:text-[#e5e5e5]"
              aria-label="Hide sidebar"
              title="Hide sidebar"
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
                      className="rounded-[6px] px-1.5 py-1 text-[11px] text-[#525252] opacity-0 transition-opacity hover:text-rose-300 group-hover:opacity-100"
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
            <p className="mb-2 px-1 text-[15px] font-bold text-[#f5f5f5]">Library</p>
            {latestLibraryDocs.map((doc) => (
              <div key={doc.doc_id} className="rounded-[8px] px-2 py-2 hover:bg-[#1a1a1a]">
                <p className="truncate text-[12.5px] font-medium text-[#d4d4d4]">{doc.filename}</p>
                <p className="text-[11px] text-[#737373]">
                  {doc.ingestion_status === "indexed" ? `${doc.chunk_count} chunks · indexed` : doc.ingestion_status}
                </p>
              </div>
            ))}
          </div>
          <div className="mt-5 border-t border-[#2a2a2a] pt-4">
            <p className="mb-2 px-1 text-[15px] font-bold text-[#f5f5f5]">Retrieval</p>
            <label className="mb-1.5 block px-1 text-[11.5px] font-medium text-[#a3a3a3]">Scope</label>
            <select value={scopeDocId} onChange={(e) => setScopeDocId(e.target.value)} className="w-full rounded-[8px] border border-[#333333] bg-[#1a1a1a] px-3 py-2 text-[12.5px] text-[#e5e5e5]">
              <option value="">All documents</option>{indexedDocs.map((d) => <option key={d.doc_id} value={d.doc_id}>{d.filename}</option>)}
            </select>
          </div>
        </aside>

        <main className="flex min-w-0 flex-col overflow-hidden bg-gradient-to-b from-[#0f0f0f] to-[#0a0a0a]">
          <div className="shrink-0 border-b border-[#2a2a2a] px-4 py-2.5">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-1">
              {!sidebarOpen && (
                <button
                  type="button"
                  onClick={() => { setSidebarOpen((v) => !v); setMenuOpen((v) => !v); }}
                  className="flex h-[34px] w-[34px] items-center justify-center rounded-[8px] text-[#a3a3a3] hover:bg-[#1f1f1f] hover:text-[#e5e5e5]"
                  aria-label="Show sidebar"
                  title="Show sidebar"
                >
                  <svg width="19" height="19" viewBox="0 0 24 24" fill="none" aria-hidden>
                    <rect x="3" y="4" width="18" height="16" rx="2.5" stroke="currentColor" strokeWidth="1.7" />
                    <line x1="9" y1="4" x2="9" y2="20" stroke="currentColor" strokeWidth="1.7" />
                  </svg>
                </button>
              )}

              <div className="relative">
                <button
                  type="button"
                  onClick={() => setHeaderMenuOpen((v) => !v)}
                  className="flex items-center gap-1.5 rounded-[8px] px-2.5 py-1.5 text-[16px] font-semibold text-[#f5f5f5] hover:bg-[#1f1f1f]"
                  aria-haspopup="menu"
                  aria-expanded={headerMenuOpen}
                >
                  Multimodal RAG
                  <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden className={`text-[#a3a3a3] transition-transform ${headerMenuOpen ? "rotate-180" : ""}`}>
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
              <div className="flex items-center gap-2">
                <span className="hidden max-w-[200px] truncate text-[12.5px] text-[#a3a3a3] sm:inline" title={user?.email}>{user?.email}</span>
                <button
                  type="button"
                  onClick={() => void logout()}
                  className="flex items-center gap-1.5 rounded-[8px] border border-[#333333] px-2.5 py-1.5 text-[12px] font-medium text-[#a3a3a3] hover:border-[#525252] hover:text-[#e5e5e5]"
                  data-testid="logout-button"
                >
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden>
                    <path d="M15 4h3a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1h-3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                    <path d="M10 8l-4 4 4 4M6 12h10" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  Log out
                </button>
              </div>
            </div>
          </div>

          {view === "chat" ? (
            <section className="flex min-h-0 flex-1 flex-col">
              <div className="flex-1 space-y-6 overflow-y-auto px-7 pb-2 pt-6">
                {messages.map((msg, idx) => (
                  <div key={msg.id ?? `${msg.role}-${idx}`} className={msg.role === "user" ? "flex justify-end" : "flex"}>
                    {msg.role === "user" ? (
                      <div className="max-w-[66%] rounded-[16px_16px_4px_16px] bg-gradient-to-br from-[#404040] to-[#2a2a2a] px-4 py-3 text-[14px] leading-relaxed text-[#f5f5f5] shadow-lg shadow-black/20 max-[880px]:max-w-[85%]">{msg.text}</div>
                    ) : (
                      <div className="max-w-[82%] max-[880px]:max-w-full">
                        <div className="rounded-[4px_16px_16px_16px] bg-gradient-to-b from-[#1f1f1f] to-[#171717] border border-[#2a2a2a] px-5 py-4 text-[15px] leading-[1.75] text-[#e5e5e5]" data-testid={`chat-msg-${idx}`}>
                          <MarkdownAnswer
                            content={msg.text}
                            placeholder={chatLoading && idx === messages.length - 1 ? "..." : ""}
                          />
                        </div>
                        {msg.sources.length > 0 && <HeroImages images={deriveHeroImages(msg.sources)} />}
                        {msg.charts.length > 0 && (
                          <ComputedChartsPanel
                            charts={msg.charts}
                            isOpen={!!openChartPanels[idx]}
                            onToggleOpen={() => setOpenChartPanels((prev) => ({ ...prev, [idx]: !prev[idx] }))}
                            messageIndex={idx}
                          />
                        )}
                        {msg.sources.length > 0 && (
                          <SourcesPanel
                            sources={msg.sources}
                            isOpen={!!openSourcePanels[idx]}
                            onToggleOpen={() => setOpenSourcePanels((prev) => ({ ...prev, [idx]: !prev[idx] }))}
                            messageIndex={idx}
                            onGoToPage={() => setView("docs")}
                            onOpenSource={(source) => openSourceInViewer(msg.sources, source)}
                          />
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
              <div className="shrink-0 border-t border-[#2a2a2a] bg-gradient-to-t from-[#0a0a0a] to-transparent px-7 pb-5 pt-4">
                {chatError && <div className="mb-2 rounded border border-rose-500/35 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">{chatError}</div>}
                <div className="mb-2 text-[12px] text-[#737373]">try — "Compare Timberland Bancorp&apos;s net interest margin across the last two fiscal years" · <span className="text-[#525252]">Enter to send, Shift+Enter for new line</span></div>
                <div className="flex items-end gap-2.5"><textarea value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={handleComposerKeyDown} placeholder="Ask a question about your documents..." className="min-h-[48px] max-h-[140px] flex-1 resize-none rounded-[14px] border border-[#333333] bg-gradient-to-b from-[#1a1a1a] to-[#141414] px-4 py-3 text-[14px] text-[#e5e5e5] outline-none placeholder:text-[#525252] focus:border-[#525252] focus:ring-1 focus:ring-[#404040]" data-testid="chat-input" /><button type="button" onClick={() => void handleSend()} disabled={!canSend} className="h-[48px] rounded-[14px] bg-gradient-to-b from-[#525252] to-[#333333] px-6 text-[13px] font-semibold text-[#f5f5f5] shadow-lg shadow-black/30 hover:from-[#737373] hover:to-[#404040] disabled:opacity-40 disabled:hover:from-[#525252] disabled:hover:to-[#333333]" data-testid="chat-send">{chatLoading ? "Streaming..." : "Send"}</button></div>
              </div>
            </section>
          ) : (
            <section className="flex-1 overflow-y-auto px-7 pb-7 pt-6">
              <h2 className="font-['Space_Grotesk'] text-[20px] font-semibold text-[#f5f5f5]">Document library</h2>
              <p className="mb-5 text-[13px] text-[#a3a3a3]">Upload PDF or DOCX files for ingestion. Status updates automatically while processing.</p>
              <div className="mb-6 rounded-[20px] border border-dashed border-[#333333] bg-gradient-to-b from-[#141414] to-[#0f0f0f] px-5 py-10 text-center hover:border-[#525252] transition-colors" onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); void handleUpload(e.dataTransfer.files?.[0]); }}>
                <input ref={fileInputRef} type="file" accept="application/pdf,.pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,.docx" className="hidden" onChange={(e) => void handleUpload(e.target.files?.[0])} />
                <p className="mb-4 text-[13.5px] text-[#a3a3a3]">Drag a PDF or DOCX here, or choose a file to index text, tables and charts for retrieval.</p>
                <button type="button" onClick={() => fileInputRef.current?.click()} className="rounded-[8px] bg-gradient-to-b from-[#525252] to-[#333333] px-5 py-2.5 text-[13px] font-semibold text-[#f5f5f5] shadow-lg shadow-black/30 hover:from-[#737373] hover:to-[#404040]">{uploading ? "Uploading..." : "Choose file"}</button>
              </div>
              {(docsError || error) && <div className="mb-3 rounded border border-rose-500/35 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">{docsError || error}</div>}
              {loading ? <p className="text-sm text-[#a3a3a3]">Loading documents...</p> : (
                <div className="space-y-2.5">
                  {documents.map((doc) => (
                    <div key={doc.doc_id} data-testid={`doc-row-${doc.doc_id}`} className="flex items-center gap-4 rounded-[14px] border border-[#2a2a2a] bg-gradient-to-b from-[#1a1a1a] to-[#141414] px-4 py-3 hover:border-[#404040] transition-colors">
                      <IngestionProgressRing
                        status={doc.ingestion_status}
                        uploadTimestamp={doc.upload_timestamp}
                        backendProgress={doc.ingestion_progress}
                        docId={doc.doc_id}
                      />
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-[13.5px] font-semibold text-[#e5e5e5]">{doc.filename}</p>
                        <p className="text-[11.5px] text-[#737373]">{formatUploadDate(doc.upload_timestamp)}</p>
                        {doc.progress_message && doc.ingestion_status !== "failed" && (
                          <p className="mt-0.5 text-[11px] text-[#a3a3a3]">{doc.progress_message}</p>
                        )}
                        {doc.error_message && (
                          <p className="mt-0.5 text-[11px] text-rose-300">{doc.error_message}</p>
                        )}
                      </div>
                      <span className={`rounded-full px-2.5 py-1 text-[11px] font-semibold ${doc.ingestion_status === "indexed" ? "bg-[#404040]/40 text-[#d4d4d4]" : isProcessing(doc.ingestion_status) ? "bg-[#525252]/30 text-[#a3a3a3]" : "bg-rose-500/15 text-rose-300"}`}>{doc.ingestion_status}</span>
                      <span className="w-10 text-right font-['Space_Grotesk'] text-[13px] font-semibold text-[#d4d4d4]">{doc.chunk_count || "—"}</span>
                      <button type="button" onClick={async () => { if (!window.confirm(`Delete "${doc.filename}" and all indexed chunks?`)) return; try { await deleteDocument(doc.doc_id); setDocuments((c) => c.filter((d) => d.doc_id !== doc.doc_id)); } catch (err) { setDocsError(err instanceof Error ? err.message : "Delete failed"); } }} className="text-xs font-medium text-[#a3a3a3] hover:text-rose-300 hover:underline">Delete</button>
                    </div>
                  ))}
                </div>
              )}
            </section>
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
    </div>
  );
}
