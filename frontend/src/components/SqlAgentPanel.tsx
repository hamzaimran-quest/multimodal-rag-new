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
  "mt-1 w-full rounded-[8px] border border-[#333333] bg-[#0f0f0f] px-3 py-2 text-[13px] text-[#e5e5e5] outline-none focus:border-[#525252]";

const textareaClass =
  "mt-1 w-full resize-y rounded-[8px] border border-[#333333] bg-[#0f0f0f] px-3 py-2.5 text-[13px] leading-relaxed text-[#e5e5e5] outline-none focus:border-[#525252] min-h-[120px] max-h-[220px] overflow-y-auto";

const descriptionPreviewClass =
  "mt-1.5 max-h-[120px] overflow-y-auto rounded-[8px] border border-[#2a2a2a] bg-[#0f0f0f] px-3 py-2 text-[12px] leading-relaxed text-[#a3a3a3] whitespace-pre-wrap";

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
  const toastIdRef = useRef(0);

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

  const connections = status?.connections ?? [];

  return (
    <section className="relative flex-1 overflow-y-auto px-7 pb-7 pt-6">
      <h2 className="font-['Space_Grotesk'] text-[20px] font-semibold text-[#f5f5f5]">SQL Agent</h2>
      <p className="mb-2 text-[13px] text-[#a3a3a3]">
        Connect read-only PostgreSQL databases. The router uses the active connection when questions need live data.
      </p>
      <p className="mb-5 text-[12px] text-[#737373]">
        Use a read-only PostgreSQL user. Credentials are encrypted at rest and never returned by the API.
      </p>

      {error && (
        <div className="mb-4 rounded border border-rose-500/35 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
          {error}
        </div>
      )}

      <div className="mb-6 rounded-[16px] border border-[#2a2a2a] bg-gradient-to-b from-[#1a1a1a] to-[#141414] p-5">
        <h3 className="mb-4 text-[14px] font-semibold text-[#e5e5e5]">Add connection</h3>
        <div className="grid gap-4">
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
            Description (schema hint for the agent)
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe tables, relationships, and business meaning. Example: DVD rental store with film, actor, category, customer, rental, payment, inventory, store, staff tables in public schema."
              rows={6}
              className={textareaClass}
              data-testid="sql-description"
            />
          </label>
        </div>
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
          <label className="flex items-center gap-2 text-[12px] text-[#a3a3a3]">
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
            className="rounded-[8px] bg-gradient-to-b from-[#525252] to-[#333333] px-4 py-2 text-[13px] font-semibold text-[#f5f5f5] disabled:opacity-40"
            data-testid="sql-add-connection"
          >
            {busyId === -1 ? "Saving..." : "Save connection"}
          </button>
        </div>
      </div>

      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h3 className="font-['Space_Grotesk'] text-[20px] font-bold text-[#f5f5f5]">Saved connections</h3>
        <button
          type="button"
          onClick={() => void runAction(0, deactivateSqlConnections)}
          disabled={!status?.has_active || busyId !== null}
          className="rounded-[8px] border border-rose-500/50 bg-rose-600/25 px-3.5 py-2 text-[12.5px] font-semibold text-rose-100 hover:bg-rose-600/35 disabled:border-[#333333] disabled:bg-transparent disabled:text-[#525252]"
          data-testid="sql-deactivate-all"
        >
          Deactivate all
        </button>
      </div>

      {loading ? (
        <p className="text-sm text-[#a3a3a3]">Loading connections...</p>
      ) : connections.length === 0 ? (
        <p className="text-sm text-[#737373]">No saved connections yet.</p>
      ) : (
        <div className="space-y-3">
          {connections.map((conn: SqlConnection) => {
            const isEditing = editingId === conn.id;
            return (
              <div
                key={conn.id}
                className={`rounded-[14px] border bg-gradient-to-b px-4 py-3 ${
                  conn.is_active
                    ? "border-emerald-500/35 from-[#1c2420] to-[#141a17]"
                    : "border-[#333333] from-[#1f1f1f] to-[#161616]"
                }`}
                data-testid={`sql-connection-${conn.id}`}
              >
                {isEditing ? (
                  <div className="grid gap-4">
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
                        className="rounded-[8px] border border-[#333333] px-3 py-1.5 text-[12px] text-[#a3a3a3] hover:border-[#525252] hover:text-[#e5e5e5] disabled:opacity-40"
                      >
                        Cancel
                      </button>
                      <button
                        type="button"
                        onClick={() => void handleSaveEdit(conn.id)}
                        disabled={busyId !== null}
                        className="rounded-[8px] bg-gradient-to-b from-[#525252] to-[#333333] px-3 py-1.5 text-[12px] font-semibold text-[#f5f5f5] disabled:opacity-40"
                        data-testid={`sql-save-edit-${conn.id}`}
                      >
                        {busyId === conn.id ? "Saving..." : "Save changes"}
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="grid gap-2.5">
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <p className="text-[13.5px] font-semibold text-[#e5e5e5]">{conn.display_name}</p>
                          {conn.is_active && (
                            <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-[10px] font-semibold text-emerald-300">
                              Active
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        <button
                          type="button"
                          onClick={() => startEdit(conn)}
                          disabled={busyId !== null || editingId !== null}
                          className="rounded-[8px] border border-[#333333] px-2.5 py-1 text-[11.5px] text-[#d4d4d4] hover:border-[#525252] disabled:opacity-40"
                          data-testid={`sql-edit-${conn.id}`}
                        >
                          Edit
                        </button>
                        {!conn.is_active && (
                          <button
                            type="button"
                            onClick={() => void runAction(conn.id, () => activateSqlConnection(conn.id))}
                            disabled={busyId !== null || editingId !== null}
                            className="rounded-[8px] border border-[#333333] px-2.5 py-1 text-[11.5px] text-[#d4d4d4] hover:border-[#525252] disabled:opacity-40"
                          >
                            Activate
                          </button>
                        )}
                        <button
                          type="button"
                          onClick={() => void handleTest(conn)}
                          disabled={busyId !== null || editingId !== null}
                          className="rounded-[8px] border border-[#333333] px-2.5 py-1 text-[11.5px] text-[#d4d4d4] hover:border-[#525252] disabled:opacity-40"
                          data-testid={`sql-test-${conn.id}`}
                        >
                          {busyId === conn.id ? "Testing..." : "Test"}
                        </button>
                        <button
                          type="button"
                          onClick={() => void runAction(conn.id, () => deleteSqlConnection(conn.id))}
                          disabled={busyId !== null || editingId !== null}
                          className="rounded-[8px] border border-[#333333] px-2.5 py-1 text-[11.5px] text-rose-300 hover:border-rose-400/40 disabled:opacity-40"
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                    <div>
                      <p className="text-[10.5px] font-medium uppercase tracking-wide text-[#737373]">Description</p>
                      <div className={descriptionPreviewClass}>{conn.description}</div>
                    </div>
                    {conn.last_error && (
                      <p className="text-[11px] text-rose-300">Last test failed: {conn.last_error}</p>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      <div className="pointer-events-none fixed bottom-5 right-5 z-50 flex w-[min(360px,calc(100vw-2.5rem))] flex-col gap-2">
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
