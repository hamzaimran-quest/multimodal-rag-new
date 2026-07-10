import { useEffect, useMemo, useState } from "react";

import { getSpreadsheetMetadata, getSpreadsheetSheet } from "../api/spreadsheet";
import { SpreadsheetGrid } from "./SpreadsheetGrid";
import type { QuerySource } from "../types";

export interface SpreadsheetViewerTarget {
  docId: string;
  filename: string;
  sources: QuerySource[];
  chunkId: string;
  sheetName?: string | null;
  sheetIndex?: number | null;
  rowRange?: number[] | null;
  colRange?: number[] | null;
}

interface SpreadsheetViewerPanelProps {
  target: SpreadsheetViewerTarget;
  onClose: () => void;
}

function citedSource(target: SpreadsheetViewerTarget): QuerySource | undefined {
  return target.sources.find((source) => source.chunk_id === target.chunkId);
}

export function SpreadsheetViewerPanel({ target, onClose }: SpreadsheetViewerPanelProps) {
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [errorMsg, setErrorMsg] = useState("");
  const [sheetNames, setSheetNames] = useState<string[]>([]);
  const [activeSheet, setActiveSheet] = useState<string | null>(target.sheetName ?? null);
  const [rows, setRows] = useState<string[][]>([]);

  const source = useMemo(() => citedSource(target), [target]);
  const rowRange = target.rowRange ?? source?.row_range ?? null;
  const colRange = target.colRange ?? source?.col_range ?? null;

  useEffect(() => {
    let active = true;
    setStatus("loading");
    getSpreadsheetMetadata(target.docId)
      .then((metadata) => {
        if (!active) return;
        const names = metadata.sheets.map((sheet) => sheet.name);
        setSheetNames(names);
        const initial =
          target.sheetName
          ?? source?.sheet_name
          ?? metadata.sheets.find((sheet) => sheet.index === (target.sheetIndex ?? source?.sheet_index))?.name
          ?? metadata.sheets[0]?.name
          ?? null;
        setActiveSheet(initial);
      })
      .catch((error: unknown) => {
        if (!active) return;
        setStatus("error");
        setErrorMsg(error instanceof Error ? error.message : "Failed to load spreadsheet");
      });
    return () => {
      active = false;
    };
  }, [target.docId, target.sheetIndex, target.sheetName, source?.sheet_index, source?.sheet_name]);

  useEffect(() => {
    if (!activeSheet) return;
    let active = true;
    setStatus("loading");
    getSpreadsheetSheet(target.docId, activeSheet)
      .then((grid) => {
        if (!active) return;
        setRows(grid.rows);
        setStatus("ready");
      })
      .catch((error: unknown) => {
        if (!active) return;
        setStatus("error");
        setErrorMsg(error instanceof Error ? error.message : "Failed to load sheet");
      });
    return () => {
      active = false;
    };
  }, [activeSheet, target.docId]);

  return (
    <div className="fixed inset-0 z-50 flex items-stretch justify-end bg-black/60 p-4" data-testid="spreadsheet-viewer">
      <div className="flex h-full w-full max-w-6xl flex-col overflow-hidden rounded-[16px] border border-[#2a2a2a] bg-[#111111] shadow-2xl">
        <div className="flex items-center justify-between border-b border-[#2a2a2a] px-4 py-3">
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-[#f5f5f5]">{target.filename}</p>
            <p className="text-xs text-[#737373]">
              Spreadsheet viewer
              {activeSheet ? ` · ${activeSheet}` : ""}
              {rowRange ? ` · rows ${rowRange[0]}–${rowRange[1]}` : ""}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-[#333333] px-3 py-1.5 text-xs text-[#d4d4d4] hover:border-[#525252]"
          >
            Close
          </button>
        </div>

        <div className="flex gap-2 overflow-x-auto border-b border-[#2a2a2a] px-4 py-2">
          {sheetNames.map((name) => (
            <button
              key={name}
              type="button"
              onClick={() => setActiveSheet(name)}
              className={`rounded-full px-3 py-1 text-xs font-medium ${
                name === activeSheet
                  ? "bg-[#404040] text-[#f5f5f5]"
                  : "bg-[#1a1a1a] text-[#a3a3a3] hover:text-[#e5e5e5]"
              }`}
            >
              {name}
            </button>
          ))}
        </div>

        <div className="min-h-0 flex-1 p-3">
          {status === "loading" && <p className="text-sm text-[#a3a3a3]">Loading sheet…</p>}
          {status === "error" && <p className="text-sm text-rose-300">{errorMsg}</p>}
          {status === "ready" && activeSheet && (
            <SpreadsheetGrid
              rows={rows}
              rowRange={rowRange}
              colRange={colRange}
              sheetKey={`${target.docId}:${activeSheet}`}
            />
          )}
        </div>
      </div>
    </div>
  );
}
