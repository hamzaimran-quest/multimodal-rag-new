import { useState } from "react";
import type { QuerySource } from "../types";
import { AuthImage } from "./AuthImage";

interface SourcesPanelProps {
  sources: QuerySource[];
  isOpen: boolean;
  onToggleOpen: () => void;
  messageIndex: number;
  onGoToPage?: (source: QuerySource) => void;
  onOpenSource?: (source: QuerySource) => void;
}

function typeDotClass(chunkType: string): string {
  if (chunkType === "image") return "bg-[#a3a3a3]";
  if (chunkType === "table") return "bg-[#737373]";
  return "bg-[#525252]";
}

function typeLabelClass(chunkType: string): string {
  if (chunkType === "image") return "text-[#d4d4d4]";
  if (chunkType === "table") return "text-[#a3a3a3]";
  return "text-[#737373]";
}

function locationLabel(source: QuerySource): string {
  if (source.source_format === "xlsx") {
    return source.sheet_name ? `sheet ${source.sheet_name}` : `sheet ${source.sheet_index ?? source.page_number}`;
  }
  if (source.source_format === "docx") {
    return source.section ? source.section : `part ${source.page_number}`;
  }
  return `p. ${source.page_number}`;
}

function openLabel(source: QuerySource): string {
  if (source.source_format === "xlsx") {
    return source.sheet_name ? `Open ${source.sheet_name}` : "Open in spreadsheet";
  }
  if (source.source_format === "docx") {
    if ((source.page_count ?? 0) > 0 && source.viewer_page) {
      return `Open page ${source.viewer_page} in document`;
    }
    return "Open in document";
  }
  return `Open page ${source.page_number} in document`;
}

function ChevronIcon({ className }: { className?: string }) {
  return (
    <svg className={className} width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden>
      <path d="M6 4L10 8L6 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function ExternalLinkIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden>
      <path d="M3 3H9V9M9 3L3 9" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
    </svg>
  );
}

export function SourcesPanel({ sources, isOpen, onToggleOpen, messageIndex, onGoToPage, onOpenSource }: SourcesPanelProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  if (sources.length === 0) return null;

  const handleRowClick = (chunkId: string) => {
    setExpandedId((current) => (current === chunkId ? null : chunkId));
  };

  const handleOpenPage = (source: QuerySource) => {
    const format = source.source_format ?? "pdf";
    const viewerReady =
      format === "xlsx"
        ? !!source.doc_id
        : format === "docx"
          ? (source.page_count ?? 0) > 0
          : !!source.doc_id;
    if (!viewerReady || !source.doc_id) {
      onGoToPage?.(source);
      return;
    }
    onOpenSource?.(source);
  };

  return (
    <div className="mt-2 w-full min-w-0 max-w-full overflow-hidden" data-testid={`sources-panel-${messageIndex}`}>
      <div className="overflow-hidden rounded-[10px] border border-[#2a2a2a] bg-gradient-to-b from-[#1a1a1a] to-[#141414]">
        <div className="flex items-center justify-between border-b border-[#2a2a2a] px-[18px] py-3.5">
          <span className="text-[12px] font-medium text-[#a3a3a3]">
            Sources · <span className="text-[#e5e5e5]">{sources.length}</span> chunks retrieved
          </span>
          <button
            type="button"
            onClick={onToggleOpen}
            className="rounded-[6px] border border-[#333333] px-2.5 py-1 text-[11px] font-medium text-[#a3a3a3] transition-colors hover:border-[#525252] hover:text-[#e5e5e5] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#525252]"
          >
            {isOpen ? "Hide Sources" : "Show Sources"}
          </button>
        </div>

        {isOpen && (
          <div className="source-list max-h-[420px] overflow-y-auto overflow-x-hidden p-1.5">
            {sources.map((source) => {
              const expanded = expandedId === source.chunk_id;
              const isImage = source.chunk_type === "image" && !!source.image_url;

              return (
                <div
                  key={source.chunk_id}
                  data-testid={`source-row-${source.chunk_id}`}
                  className={`mb-1 rounded-[8px] border border-transparent transition-colors last:mb-0 ${expanded ? "border-[#333333] bg-[#1f1f1f]" : ""}`}
                >
                  <button
                    type="button"
                    aria-expanded={expanded}
                    onClick={() => handleRowClick(source.chunk_id)}
                    className="flex w-full cursor-pointer items-center gap-3 rounded-[8px] px-3 py-2.5 text-left transition-colors hover:bg-[#262626] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-[#525252]"
                  >
                    {isImage ? (
                      <AuthImage
                        src={source.image_url ?? ""}
                        alt=""
                        className="h-10 w-10 shrink-0 rounded-[6px] border border-[#333333] object-cover"
                      />
                    ) : (
                      <span className={`h-2 w-2 shrink-0 rounded-full ${typeDotClass(source.chunk_type)}`} />
                    )}

                    <div className="min-w-0 flex-1">
                      <div className="truncate text-[13.5px] font-medium text-[#f5f5f5]">{source.filename}</div>
                      <div className="mt-px text-[11px] text-[#737373]">
                        {locationLabel(source)} ·{" "}
                        <span className={typeLabelClass(source.chunk_type)}>
                          {source.chunk_type}
                        </span>
                      </div>
                    </div>

                    <ChevronIcon
                      className={`shrink-0 text-[#525252] transition-transform duration-200 ${expanded ? "rotate-90 text-[#d4d4d4]" : ""}`}
                    />
                  </button>

                  <div className={`source-row-body ${expanded ? "expanded" : ""}`}>
                    <div className="overflow-hidden">
                      <div className="px-3.5 pb-3.5 pl-[34px]">
                        {isImage ? (
                          <>
                            {source.snippet ? (
                              <p className="break-words border-l-2 border-[#333333] pl-3 text-[13px] leading-[1.55] text-[#a3a3a3]">
                                {source.snippet}
                              </p>
                            ) : (
                              <p className="break-words border-l-2 border-[#333333] pl-3 text-[13px] leading-[1.55] text-[#a3a3a3]">
                                Extracted image · {source.filename} · page {source.page_number}
                              </p>
                            )}
                            <AuthImage
                              src={source.image_url ?? ""}
                              alt={`Image from ${source.filename}, page ${source.page_number}`}
                              className="mt-2.5 h-20 w-auto max-w-[140px] rounded-[6px] border border-[#333333] object-contain"
                            />
                          </>
                        ) : (
                          <p className="break-words border-l-2 border-[#333333] pl-3 text-[13px] leading-[1.55] text-[#a3a3a3]">
                            {source.snippet}
                          </p>
                        )}

                        {(source.attached_images?.length ?? 0) > 0 && (
                          <div className="mt-2.5 flex flex-wrap gap-2">
                            {source.attached_images!.map((image) => (
                              <AuthImage
                                key={image.image_chunk_id}
                                src={image.image_url}
                                alt={image.caption || "Related image"}
                                className="h-16 w-auto max-w-[120px] rounded-[6px] border border-[#333333] object-contain"
                              />
                            ))}
                          </div>
                        )}

                        <button
                          type="button"
                          onClick={() => handleOpenPage(source)}
                          className="mt-2.5 inline-flex items-center gap-1.5 border-none bg-transparent p-0 text-[12px] text-[#d4d4d4] hover:underline"
                        >
                          <ExternalLinkIcon />
                          {openLabel(source)}
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
