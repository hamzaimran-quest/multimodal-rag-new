import type { ComputedChart, QuerySource, SqlMeta } from "../types";
import { authFetch } from "./http";

type QueryStreamEvent =
  | { event: "meta"; data: Record<string, unknown> }
  | { event: "route"; data: { mode: "sql" | "rag" | "hybrid" } }
  | { event: "tool"; data: { name: string; status: "running" | "complete"; round?: number; label?: string } }
  | { event: "token"; data: { token: string } }
  | { event: "sql"; data: SqlMeta }
  | { event: "sources"; data: { sources: QuerySource[] } }
  | { event: "charts"; data: { charts: ComputedChart[] } }
  | { event: "done"; data: { ok: boolean } }
  | { event: "error"; data: { message: string } };

const TOOL_LABELS: Record<string, string> = {
  search_documents: "Searching documents",
  search_images: "Finding images",
  list_documents: "Listing documents",
  create_chart: "Creating chart",
  query_database: "Querying database",
};

export function parseSseChunk(chunk: string): QueryStreamEvent[] {
  const events: QueryStreamEvent[] = [];
  const frames = chunk.split("\n\n").map((f) => f.trim()).filter(Boolean);

  for (const frame of frames) {
    const lines = frame.split("\n");
    const eventLine = lines.find((l) => l.startsWith("event: "));
    const dataLine = lines.find((l) => l.startsWith("data: "));
    if (!eventLine || !dataLine) continue;

    const event = eventLine.slice(7).trim() as QueryStreamEvent["event"];
    try {
      const data = JSON.parse(dataLine.slice(6).trim()) as QueryStreamEvent["data"];
      events.push({ event, data } as QueryStreamEvent);
    } catch {
      // ignore malformed event payloads
    }
  }
  return events;
}

export { TOOL_LABELS };

export async function streamQuery(
  payload: {
    query: string;
    doc_id?: string | null;
    doc_ids?: string[] | null;
    scope_empty?: boolean;
    session_id?: number | null;
  },
  handlers: {
    onMeta?: (meta: { session_id?: number }) => void;
    onTool?: (tool: { name: string; status: "running" | "complete"; round?: number; label?: string }) => void;
    onToken: (token: string) => void;
    onSources: (sources: QuerySource[]) => void;
    onCharts?: (charts: ComputedChart[]) => void;
    onSql?: (sql: SqlMeta) => void;
    onRoute?: (mode: "sql" | "rag" | "hybrid") => void;
    onError: (message: string) => void;
  },
): Promise<void> {
  const response = await authFetch("/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: payload.query,
      doc_id: payload.doc_id ?? undefined,
      doc_ids: payload.doc_ids ?? undefined,
      scope_empty: payload.scope_empty ?? undefined,
      session_id: payload.session_id ?? undefined,
    }),
  });

  if (!response.ok || !response.body) {
    throw new Error(`Stream failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";

    for (const part of parts) {
      const events = parseSseChunk(part + "\n\n");
      for (const event of events) {
        if (event.event === "meta") handlers.onMeta?.(event.data as { session_id?: number });
        if (event.event === "route") handlers.onRoute?.(event.data.mode);
        if (event.event === "tool") handlers.onTool?.(event.data);
        if (event.event === "token") handlers.onToken(event.data.token);
        if (event.event === "sql") handlers.onSql?.(event.data);
        if (event.event === "sources") handlers.onSources(event.data.sources);
        if (event.event === "charts") handlers.onCharts?.(event.data.charts);
        if (event.event === "error") handlers.onError(event.data.message);
      }
    }
  }
}
