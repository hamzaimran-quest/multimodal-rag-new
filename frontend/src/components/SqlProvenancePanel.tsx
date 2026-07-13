import type { SqlMeta } from "../types";

interface SqlProvenancePanelProps {
  sqlMeta: SqlMeta;
  isOpen: boolean;
  onToggleOpen: () => void;
  messageIndex: number;
}

function ChevronIcon({ className }: { className?: string }) {
  return (
    <svg className={className} width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden>
      <path d="M6 4L10 8L6 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function routeBadge(mode?: string): string {
  if (mode === "hybrid") return "Database + documents";
  if (mode === "sql") return "Database";
  return "Database";
}

export function SqlProvenancePanel({ sqlMeta, isOpen, onToggleOpen, messageIndex }: SqlProvenancePanelProps) {
  const queryCount = sqlMeta.queries?.length ?? 0;

  return (
    <div className="rounded-[10px] border border-[#2a2a2a] bg-[#141414]" data-testid={`sql-provenance-${messageIndex}`}>
      <button
        type="button"
        onClick={onToggleOpen}
        className="flex w-full items-center gap-2 px-3 py-2.5 text-left hover:bg-[#1a1a1a]"
        aria-expanded={isOpen}
      >
        <ChevronIcon className={`shrink-0 text-[#737373] transition-transform ${isOpen ? "rotate-90" : ""}`} />
        <span className="text-[12.5px] font-medium text-[#d4d4d4]">
          {routeBadge(sqlMeta.route_mode)} · {sqlMeta.display_name}
        </span>
        <span className="ml-auto rounded-full bg-[#262626] px-2 py-0.5 text-[10.5px] text-[#a3a3a3]">
          {queryCount} quer{queryCount === 1 ? "y" : "ies"}
        </span>
      </button>
      {isOpen && (
        <div className="space-y-2 border-t border-[#2a2a2a] px-3 py-3">
          {(sqlMeta.queries ?? []).map((query, idx) => (
            <pre
              key={`${idx}-${query.slice(0, 24)}`}
              className="overflow-x-auto rounded-[8px] border border-[#333333] bg-[#0f0f0f] px-3 py-2 font-mono text-[11.5px] leading-relaxed text-[#d4d4d4]"
            >
              {query}
            </pre>
          ))}
          {queryCount === 0 && (
            <p className="text-[12px] text-[#737373]">No SQL statements were recorded for this reply.</p>
          )}
        </div>
      )}
    </div>
  );
}
