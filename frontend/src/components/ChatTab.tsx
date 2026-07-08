import { useMemo, useState } from "react";

import { streamQuery } from "../api/query";
import type { QuerySource } from "../types";
import { AuthImage } from "./AuthImage";

interface ChatMessage {
  role: "user" | "assistant";
  text: string;
  sources?: QuerySource[];
}

export function ChatTab() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  const canSend = query.trim().length > 0 && !loading;

  const latestAssistantIndex = useMemo(
    () =>
      [...messages]
        .map((m, i) => ({ m, i }))
        .reverse()
        .find((x) => x.m.role === "assistant")?.i,
    [messages],
  );

  const onSend = async () => {
    const prompt = query.trim();
    if (!prompt || loading) return;

    setError(null);
    setLoading(true);
    setQuery("");
    setMessages((prev) => [...prev, { role: "user", text: prompt }, { role: "assistant", text: "" }]);

    try {
      await streamQuery(
        { query: prompt },
        {
          onToken: (token) => {
            setMessages((prev) => {
              const next = [...prev];
              const idx = [...next]
                .map((m, i) => ({ m, i }))
                .reverse()
                .find((x) => x.m.role === "assistant")?.i;
              if (idx === undefined) return next;
              next[idx] = { ...next[idx], text: next[idx].text + token };
              return next;
            });
          },
          onSources: (sources) => {
            setMessages((prev) => {
              const next = [...prev];
              const idx = [...next]
                .map((m, i) => ({ m, i }))
                .reverse()
                .find((x) => x.m.role === "assistant")?.i;
              if (idx === undefined) return next;
              next[idx] = { ...next[idx], sources };
              return next;
            });
          },
          onError: (message) => setError(message),
        },
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Streaming failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
        <div className="space-y-3">
          {messages.length === 0 ? (
            <p className="text-sm text-slate-500">
              Ask about your uploaded documents (e.g. "Summarize the key points on page 2").
            </p>
          ) : (
            messages.map((msg, idx) => (
              <div
                key={`${msg.role}-${idx}`}
                className={`rounded-lg px-3 py-2 text-sm ${
                  msg.role === "user"
                    ? "bg-indigo-500/20 text-indigo-100"
                    : "bg-slate-800/80 text-slate-100"
                }`}
                data-testid={`chat-msg-${idx}`}
              >
                <p className="whitespace-pre-wrap">{msg.text || (loading && idx === latestAssistantIndex ? "..." : "")}</p>
                {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (
                  <div className="mt-3 space-y-2 border-t border-slate-700 pt-2">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Sources</p>
                    {msg.sources.map((source) => (
                      <div key={source.chunk_id} className="rounded-md border border-slate-700 bg-slate-900/60 p-2 text-xs">
                        <p className="text-slate-300">
                          {source.filename} · {source.source_format === "docx" ? (source.section ?? `part ${source.page_number}`) : `page ${source.page_number}`} · {source.chunk_type}
                        </p>
                        <p className="mt-1 text-slate-400">{source.snippet}</p>
                        {source.image_url && (
                          <AuthImage
                            src={source.image_url}
                            alt={`${source.filename} page ${source.page_number}`}
                            className="mt-2 max-h-40 rounded border border-slate-700 object-contain"
                          />
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-2 text-sm text-rose-200">
          {error}
        </div>
      )}

      <div className="flex items-end gap-2">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about your documents..."
          className="min-h-[84px] flex-1 rounded-lg border border-slate-700 bg-slate-900/60 px-3 py-2 text-sm text-slate-100 outline-none ring-indigo-500 placeholder:text-slate-500 focus:ring-2"
          data-testid="chat-input"
        />
        <button
          type="button"
          onClick={() => void onSend()}
          disabled={!canSend}
          className="rounded-lg bg-indigo-500 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-60"
          data-testid="chat-send"
        >
          {loading ? "Streaming..." : "Send"}
        </button>
      </div>
    </div>
  );
}

