import { useCallback, useEffect, useRef, useState } from "react";

import {
  activateSqlConnection,
  addSqlConnection,
  deactivateSqlConnections,
  deleteSqlConnection,
  getSqlAgentStatus,
  testSqlConnection,
  updateSqlConnection,
  updateSqlCredentials,
} from "../api/sqlAgent";
import type { SqlAgentStatus, SqlConnection } from "../types";

interface SqlAgentPanelProps {
  onStatusChange?: (status: SqlAgentStatus) => void;
}

interface Toast {
  id: number;
  type: "success" | "error";
  message: string;
}

interface EditForm {
  display_name: string;
  description: string;
  connection_url: string;
}

const inputClass =
  "mt-1 w-full rounded-[8px] border border-[#333333] bg-[#0f0f0f] px-3 py-2 text-[13px] text-[#e5e5e5] outline-none focus:border-[#525252] max-[880px]:text-[16px] max-[880px]:py-2.5";

const textareaClass =
  "mt-1 w-full resize-y rounded-[8px] border border-[#333333] bg-[#0f0f0f] px-3 py-2.5 text-[13px] leading-relaxed text-[#e5e5e5] outline-none focus:border-[#525252] min-h-[120px] max-h-[220px] overflow-y-auto max-[880px]:min-h-[96px] max-[880px]:max-h-[160px] max-[880px]:text-[16px]";

const descriptionPreviewClass =
  "mt-1.5 max-h-[120px] overflow-y-auto rounded-[8px] border border-[#2a2a2a] bg-[#0f0f0f] px-3 py-2 text-[12px] leading-relaxed text-[#a3a3a3] whitespace-pre-wrap max-[880px]:max-h-[88px]";

const actionBtnClass =
  "rounded-[8px] border border-[#333333] px-2.5 py-1 text-[11.5px] text-[#d4d4d4] hover:border-[#525252] disabled:opacity-40 max-[880px]:min-h-10 max-[880px]:flex-1 max-[880px]:px-2 max-[880px]:text-[12px]";

export function SqlAgentPanel({ onStatusChange }: SqlAgentPanelProps) {
  const [status, setStatus] = useState<SqlAgentStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<number | null>(null);
  const [connectionUrl, setConnectionUrl] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [description, setDescription] = useState("");
  const [activateOnAdd, setActivateOnAdd] = useState(true);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editForm, setEditForm] = useState<EditForm>({ display_name: "", description: "", connection_url: "" });
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [addOpen, setAddOpen] = useState(true);
  const [expandedDescIds, setExpandedDescIds] = useState<Set<number>>(new Set());
  const toastIdRef = useRef(0);

  useEffect(() => {
    const mq = window.matchMedia("(max-width: 880px)");
    const sync = () => setAddOpen(!mq.matches);
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, []);

  const pushToast = useCallback((type: Toast["type"], message: string) => {
    const id = ++toastIdRef.current;
    setToasts((current) => [...current, { id, type, message }]);
    window.setTimeout(() => {
      setToasts((current) => current.filter((toast) => toast.id !== id));
    }, 4000);
  }, []);

  const refresh = useCallback(async () => {
    setError(null);
    try {
      const next = await getSqlAgentStatus();
      setStatus(next);
      onStatusChange?.(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load SQL agent status");
    } finally {
      setLoading(false);
    }
  }, [onStatusChange]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const runAction = async (connectionId: number, action: () => Promise<unknown>) => {
    setBusyId(connectionId);
    setError(null);
    try {
      await action();
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Action failed");
    } finally {
      setBusyId(null);
    }
  };

  const handleTest = async (conn: SqlConnection) => {
    setBusyId(conn.id);
    setError(null);
    try {
      await testSqlConnection(conn.id);
      await refresh();
      pushToast("success", `Connection test passed for "${conn.display_name}".`);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Connection test failed";
      await refresh();
      pushToast("error", `Connection test failed for "${conn.display_name}".`);
      setError(message);
    } finally {
      setBusyId(null);
    }
  };

  const handleAdd = async () => {
    if (!connectionUrl.trim() || !displayName.trim() || !description.trim()) {
      setError("Connection URL, display name, and description are required.");
      return;
    }
    setBusyId(-1);
    setError(null);
    try {
      await addSqlConnection({
        connection_url: connectionUrl.trim(),
        display_name: displayName.trim(),
        description: description.trim(),
        activate: activateOnAdd,
      });
      setConnectionUrl("");
      setDisplayName("");
      setDescription("");
      await refresh();
      pushToast("success", "Connection saved.");
      if (window.matchMedia("(max-width: 880px)").matches) setAddOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add connection");
    } finally {
      setBusyId(null);
    }
  };

  const startEdit = (conn: SqlConnection) => {
    setEditingId(conn.id);
    setEditForm({
      display_name: conn.display_name,
      description: conn.description,
      connection_url: "",
    });
    setError(null);
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditForm({ display_name: "", description: "", connection_url: "" });
  };

  const handleSaveEdit = async (connectionId: number) => {
    if (!editForm.display_name.trim() || !editForm.description.trim()) {
      setError("Display name and description are required.");
      return;
    }
    setBusyId(connectionId);
    setError(null);
    try {
      await updateSqlConnection(connectionId, {
        display_name: editForm.display_name.trim(),
        description: editForm.description.trim(),
      });
      if (editForm.connection_url.trim()) {
        await updateSqlCredentials(connectionId, editForm.connection_url.trim());
      }
      setEditingId(null);
      setEditForm({ display_name: "", description: "", connection_url: "" });
      await refresh();
      pushToast("success", "Connection updated.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update connection");
    } finally {
      setBusyId(null);
    }
  };

  const toggleDesc = (id: number) => {
    setExpandedDescIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const connections = status?.connections ?? [];

  return (
    <section className="relative flex-1 overflow-y-auto px-7 pb-7 pt-6 max-[880px]:px-3 max-[880px]:pb-6 max-[880px]:pt-4">
      <h2 className="font-['Space_Grotesk'] text-[20px] font-semibold text-[#f5f5f5] max-[880px]:text-[18px]">SQL Agent</h2>
      <p className="mb-5 text-[13px] text-[#a3a3a3] max-[880px]:mb-4 max-[880px]:text-[12.5px]">
        <span className="max-[880px]:hidden">
          Connect read-only PostgreSQL databases. The router uses the active connection when questions need live data.
          Credentials are encrypted at rest and never returned by the API.
        </span>
        <span className="hidden max-[880px]:inline">Read-only PostgreSQL connections for live data questions.</span>
      </p>

      {error && (
        <div className="mb-4 rounded border border-rose-500/35 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
          {error}
        </div>
      )}

      <div className="mb-6 rounded-[16px] border border-[#2a2a2a] bg-gradient-to-b from-[#1a1a1a] to-[#141414] max-[880px]:mb-4">
        <button
          type="button"
          onClick={() => setAddOpen((open) => !open)}
          className="flex w-full items-center justify-between gap-3 px-5 py-4 text-left max-[880px]:min-h-12 max-[880px]:px-3.5 max-[880px]:py-3.5 min-[881px]:pointer-events-none"
          aria-expanded={addOpen}
          data-testid="sql-add-toggle"
        >
          <h3 className="text-[14px] font-semibold text-[#e5e5e5]">Add connection</h3>
          <span className="text-[12px] text-[#737373] min-[881px]:hidden">{addOpen ? "Hide" : "Show"}</span>
        </button>
        <div className={`${addOpen ? "block" : "hidden"} px-5 pb-5 max-[880px]:px-3.5 max-[880px]:pb-4 min-[881px]:block`}>
          <div className="grid gap-4 max-[880px]:gap-3">
            <label className="block text-[12px] font-medium text-[#a3a3a3]">
              PostgreSQL URL
              <input
                value={connectionUrl}
                onChange={(e) => setConnectionUrl(e.target.value)}
                placeholder="postgresql://readonly:pass@host:5432/analytics"
                className={inputClass}
                data-testid="sql-connection-url"
              />
            </label>
            <label className="block text-[12px] font-medium text-[#a3a3a3]">
              Display name
              <input
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder="Analytics DB"
                className={inputClass}
                data-testid="sql-display-name"
              />
            </label>
            <label className="block text-[12px] font-medium text-[#a3a3a3]">
              Description
              <span className="font-normal text-[#737373] max-[880px]:hidden"> (schema hint for the agent)</span>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Tables, relationships, and business meaning…"
                rows={6}
                className={textareaClass}
                data-testid="sql-description"
              />
            </label>
          </div>
          <div className="mt-4 flex flex-wrap items-center justify-between gap-3 max-[880px]:mt-3 max-[880px]:flex-col max-[880px]:items-stretch">
            <label className="flex items-center gap-2 text-[12px] text-[#a3a3a3] max-[880px]:min-h-10">
              <input
                type="checkbox"
                checked={activateOnAdd}
                onChange={(e) => setActivateOnAdd(e.target.checked)}
                className="accent-[#d4d4d4]"
              />
              Set as active connection
            </label>
            <button
              type="button"
              onClick={() => void handleAdd()}
              disabled={busyId !== null}
              className="rounded-[8px] bg-gradient-to-b from-[#525252] to-[#333333] px-4 py-2 text-[13px] font-semibold text-[#f5f5f5] disabled:opacity-40 max-[880px]:min-h-11"
              data-testid="sql-add-connection"
            >
              {busyId === -1 ? "Saving..." : "Save connection"}
            </button>
          </div>
        </div>
      </div>

      <div className="mb-4 flex items-center justify-between gap-3 max-[880px]:mb-3">
        <h3 className="font-['Space_Grotesk'] text-[16px] font-semibold text-[#f5f5f5] max-[880px]:text-[15px]">
          Connections
          {!loading && connections.length > 0 && (
            <span className="ml-1.5 text-[13px] font-normal text-[#737373]">{connections.length}</span>
          )}
        </h3>
        <button
          type="button"
          onClick={() => void runAction(0, deactivateSqlConnections)}
          disabled={!status?.has_active || busyId !== null}
          className="rounded-[8px] border border-rose-500/50 bg-rose-600/25 px-3.5 py-2 text-[12.5px] font-semibold text-rose-100 hover:bg-rose-600/35 disabled:border-[#333333] disabled:bg-transparent disabled:text-[#525252] max-[880px]:min-h-10 max-[880px]:px-2.5 max-[880px]:text-[11.5px]"
          data-testid="sql-deactivate-all"
        >
          <span className="max-[880px]:hidden">Deactivate all</span>
          <span className="hidden max-[880px]:inline">Deactivate</span>
        </button>
      </div>

      {loading ? (
        <p className="text-sm text-[#a3a3a3]">Loading connections...</p>
      ) : connections.length === 0 ? (
        <p className="text-sm text-[#737373]">
          No saved connections yet.
          <button type="button" onClick={() => setAddOpen(true)} className="ml-1 text-[#d4d4d4] underline max-[880px]:inline min-[881px]:hidden">
            Add one
          </button>
        </p>
      ) : (
        <div className="space-y-3 max-[880px]:space-y-2.5">
          {connections.map((conn: SqlConnection) => {
            const isEditing = editingId === conn.id;
            const descOpen = expandedDescIds.has(conn.id);
            return (
              <div
                key={conn.id}
                className={`rounded-[14px] border bg-gradient-to-b px-4 py-3 max-[880px]:px-3 max-[880px]:py-2.5 ${
                  conn.is_active
                    ? "border-emerald-500/35 from-[#1c2420] to-[#141a17]"
                    : "border-[#333333] from-[#1f1f1f] to-[#161616]"
                }`}
                data-testid={`sql-connection-${conn.id}`}
              >
                {isEditing ? (
                  <div className="grid gap-4 max-[880px]:gap-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <p className="text-[13px] font-semibold text-[#e5e5e5]">Edit connection</p>
                      {conn.is_active && (
                        <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-[10.5px] font-semibold text-emerald-300">
                          Active
                        </span>
                      )}
                    </div>
                    <label className="block text-[12px] font-medium text-[#a3a3a3]">
                      Display name
                      <input
                        value={editForm.display_name}
                        onChange={(e) => setEditForm((prev) => ({ ...prev, display_name: e.target.value }))}
                        className={inputClass}
                        data-testid={`sql-edit-name-${conn.id}`}
                      />
                    </label>
                    <label className="block text-[12px] font-medium text-[#a3a3a3]">
                      Description
                      <textarea
                        value={editForm.description}
                        onChange={(e) => setEditForm((prev) => ({ ...prev, description: e.target.value }))}
                        rows={6}
                        className={textareaClass}
                        data-testid={`sql-edit-description-${conn.id}`}
                      />
                    </label>
                    <label className="block text-[12px] font-medium text-[#a3a3a3]">
                      New PostgreSQL URL (optional)
                      <input
                        value={editForm.connection_url}
                        onChange={(e) => setEditForm((prev) => ({ ...prev, connection_url: e.target.value }))}
                        placeholder="Leave blank to keep current credentials"
                        className={inputClass}
                        data-testid={`sql-edit-url-${conn.id}`}
                      />
                    </label>
                    <div className="flex flex-wrap justify-end gap-2">
                      <button
                        type="button"
                        onClick={cancelEdit}
                        disabled={busyId !== null}
                        className="rounded-[8px] border border-[#333333] px-3 py-1.5 text-[12px] text-[#a3a3a3] hover:border-[#525252] hover:text-[#e5e5e5] disabled:opacity-40 max-[880px]:min-h-10 max-[880px]:flex-1"
                      >
                        Cancel
                      </button>
                      <button
                        type="button"
                        onClick={() => void handleSaveEdit(conn.id)}
                        disabled={busyId !== null}
                        className="rounded-[8px] bg-gradient-to-b from-[#525252] to-[#333333] px-3 py-1.5 text-[12px] font-semibold text-[#f5f5f5] disabled:opacity-40 max-[880px]:min-h-10 max-[880px]:flex-1"
                        data-testid={`sql-save-edit-${conn.id}`}
                      >
                        {busyId === conn.id ? "Saving..." : "Save changes"}
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="grid gap-2.5 max-[880px]:gap-2">
                    <div className="flex min-w-0 items-center gap-2">
                      <p className="truncate text-[13.5px] font-semibold text-[#e5e5e5] max-[880px]:text-[13px]">{conn.display_name}</p>
                      {conn.is_active && (
                        <span className="shrink-0 rounded-full bg-emerald-500/15 px-2 py-0.5 text-[10px] font-semibold text-emerald-300">
                          Active
                        </span>
                      )}
                    </div>

                    <button
                      type="button"
                      onClick={() => toggleDesc(conn.id)}
                      className="hidden w-full text-left text-[12px] text-[#737373] max-[880px]:block"
                      aria-expanded={descOpen}
                    >
                      {descOpen ? "Hide description" : "Show description"}
                    </button>
                    <div className={descOpen ? "block" : "max-[880px]:hidden"}>
                      <p className="text-[10.5px] font-medium uppercase tracking-wide text-[#737373] max-[880px]:hidden">Description</p>
                      <div className={descriptionPreviewClass}>{conn.description}</div>
                    </div>

                    {conn.last_error && (
                      <p className="line-clamp-2 text-[11px] text-rose-300">Last test failed: {conn.last_error}</p>
                    )}

                    <div className="flex flex-wrap gap-2 max-[880px]:gap-1.5">
                      <button type="button" onClick={() => startEdit(conn)} disabled={busyId !== null || editingId !== null} className={actionBtnClass} data-testid={`sql-edit-${conn.id}`}>
                        Edit
                      </button>
                      {!conn.is_active && (
                        <button type="button" onClick={() => void runAction(conn.id, () => activateSqlConnection(conn.id))} disabled={busyId !== null || editingId !== null} className={actionBtnClass}>
                          Activate
                        </button>
                      )}
                      <button type="button" onClick={() => void handleTest(conn)} disabled={busyId !== null || editingId !== null} className={actionBtnClass} data-testid={`sql-test-${conn.id}`}>
                        {busyId === conn.id ? "Testing…" : "Test"}
                      </button>
                      <button type="button" onClick={() => void runAction(conn.id, () => deleteSqlConnection(conn.id))} disabled={busyId !== null || editingId !== null} className={`${actionBtnClass} text-rose-300 hover:border-rose-400/40`}>
                        Delete
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      <div className="pointer-events-none fixed bottom-5 right-5 z-50 flex w-[min(360px,calc(100vw-2.5rem))] flex-col gap-2 max-[880px]:bottom-[max(1rem,env(safe-area-inset-bottom))] max-[880px]:right-3 max-[880px]:left-3 max-[880px]:w-auto">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={`pointer-events-auto rounded-[12px] border px-4 py-3 text-[13px] shadow-2xl shadow-black/40 ${
              toast.type === "success"
                ? "border-emerald-500/35 bg-[#102018] text-emerald-200"
                : "border-rose-500/35 bg-[#1a1012] text-rose-200"
            }`}
            data-testid={`sql-toast-${toast.type}`}
          >
            {toast.message}
          </div>
        ))}
      </div>
    </section>
  );
}
