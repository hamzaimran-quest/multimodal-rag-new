import { useEffect, useRef, useState } from "react";

import { getEmbeddingModelStatus } from "../api/client";

const POLL_INTERVAL_MS = 1500;
const ERROR_DISPLAY_MS = 6000;

/** Top-right banner shown while the backend's embedding model warms up in the
 * background (see app.ingestion.embeddings.start_embedding_model_warmup).
 * Polls status and disappears once the model is ready; document/query
 * requests that need embeddings before then still work, they just block
 * briefly on the backend's model lock rather than failing. */
export function EmbeddingModelBanner() {
  const [status, setStatus] = useState<"checking" | "loading" | "ready" | "error">("checking");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const startedAtRef = useRef<number | null>(null);
  const dismissedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    let pollTimer: ReturnType<typeof setTimeout> | null = null;
    let tickTimer: ReturnType<typeof setInterval> | null = null;

    const poll = async () => {
      if (cancelled || dismissedRef.current) return;
      try {
        const result = await getEmbeddingModelStatus();
        if (cancelled) return;

        if (result.state === "loading") {
          startedAtRef.current = result.started_at;
          setStatus("loading");
          pollTimer = setTimeout(poll, POLL_INTERVAL_MS);
          return;
        }
        if (result.state === "error") {
          setStatus("error");
          dismissedRef.current = true;
          window.setTimeout(() => {
            if (!cancelled) setStatus("ready");
          }, ERROR_DISPLAY_MS);
          return;
        }
        // "ready" or "not_started" (warm-up never needed / already done elsewhere).
        setStatus("ready");
      } catch {
        // Status endpoint unreachable -- don't block the UI on it; just stop polling.
        setStatus("ready");
      }
    };

    void poll();

    tickTimer = setInterval(() => {
      if (startedAtRef.current) {
        setElapsedSeconds(Math.max(0, Math.round(Date.now() / 1000 - startedAtRef.current)));
      }
    }, 1000);

    return () => {
      cancelled = true;
      if (pollTimer) clearTimeout(pollTimer);
      if (tickTimer) clearInterval(tickTimer);
    };
  }, []);

  if (status === "ready" || status === "checking") return null;

  return (
    <div
      className="fixed right-4 top-4 z-[60] flex items-center gap-2.5 rounded-[10px] border border-[#2a2a2a] bg-gradient-to-b from-[#1a1a1a] to-[#141414] px-3.5 py-2.5 text-[12px] text-[#e5e5e5] shadow-lg shadow-black/30"
      data-testid="embedding-model-banner"
      role="status"
    >
      {status === "loading" ? (
        <>
          <span className="h-3 w-3 shrink-0 animate-spin rounded-full border-2 border-[#525252] border-t-[#e5e5e5]" />
          <span>
            Loading embedding model{elapsedSeconds > 0 ? ` · ${elapsedSeconds}s` : ""}
          </span>
        </>
      ) : (
        <span className="text-[#f5a3a3]">Model warm-up failed — will load on first use</span>
      )}
    </div>
  );
}
