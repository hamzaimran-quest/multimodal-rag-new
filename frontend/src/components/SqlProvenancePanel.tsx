import type { SqlMeta } from "../types";

interface SqlProvenancePanelProps {
  sqlMeta: SqlMeta;
  messageIndex: number;
}

function routeBadge(mode?: string): string {
  if (mode === "hybrid") return "Database + documents";
  return "Database";
}

export function SqlProvenancePanel({ sqlMeta, messageIndex }: SqlProvenancePanelProps) {
  return (
    <div
      className="rounded-[10px] border border-[#2a2a2a] bg-[#141414] px-3 py-2.5"
      data-testid={`sql-provenance-${messageIndex}`}
    >
      <span className="text-[12.5px] font-medium text-[#d4d4d4]">
        {routeBadge(sqlMeta.route_mode)} · {sqlMeta.display_name}
      </span>
    </div>
  );
}
