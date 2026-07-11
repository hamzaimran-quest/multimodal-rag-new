import { useMemo, useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";

import { columnLabel } from "../lib/spreadsheet";

const ROW_HEIGHT = 32;
const ROW_NUMBER_WIDTH = 48;
const COLUMN_WIDTH = 120;

interface SpreadsheetGridProps {
  rows: string[][];
  sheetKey: string;
}

export function SpreadsheetGrid({ rows, sheetKey }: SpreadsheetGridProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const colCount = useMemo(
    () => rows.reduce((max, row) => Math.max(max, row.length), 0),
    [rows],
  );

  const rowVirtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 12,
  });

  if (rows.length === 0) {
    return <p className="text-sm text-[#a3a3a3]">This sheet is empty.</p>;
  }

  const tableWidth = ROW_NUMBER_WIDTH + colCount * COLUMN_WIDTH;

  return (
    <div ref={scrollRef} className="h-full overflow-auto" data-testid="spreadsheet-grid">
      <div style={{ width: tableWidth, position: "relative" }}>
        <div
          className="sticky top-0 z-20 flex border-b border-[#2a2a2a] bg-[#1a1a1a] text-[11px] font-semibold uppercase tracking-wide text-[#a3a3a3]"
          style={{ height: ROW_HEIGHT }}
        >
          <div
            className="sticky left-0 z-30 flex shrink-0 items-center justify-center border-r border-[#2a2a2a] bg-[#1a1a1a]"
            style={{ width: ROW_NUMBER_WIDTH }}
          >
            #
          </div>
          {Array.from({ length: colCount }, (_, colIndex) => (
            <div
              key={`header-${colIndex}`}
              className="flex shrink-0 items-center border-r border-[#2a2a2a] px-2"
              style={{ width: COLUMN_WIDTH }}
            >
              {columnLabel(colIndex + 1)}
            </div>
          ))}
        </div>

        <div style={{ height: rowVirtualizer.getTotalSize(), position: "relative" }}>
          {rowVirtualizer.getVirtualItems().map((virtualRow) => {
            const rowNumber = virtualRow.index + 1;
            const row = rows[virtualRow.index] ?? [];
            return (
              <div
                key={`${sheetKey}-${virtualRow.key}`}
                data-row-number={rowNumber}
                className="absolute left-0 flex text-xs text-[#e5e5e5]"
                style={{
                  top: virtualRow.start,
                  height: virtualRow.size,
                  width: tableWidth,
                }}
              >
                <div
                  className="sticky left-0 z-10 flex shrink-0 items-center justify-center border-r border-b border-[#2a2a2a] bg-[#141414] text-[11px] text-[#737373]"
                  style={{ width: ROW_NUMBER_WIDTH, height: ROW_HEIGHT }}
                >
                  {rowNumber}
                </div>
                {Array.from({ length: colCount }, (_, colIndex) => {
                  const colNumber = colIndex + 1;
                  const value = row[colIndex] ?? "";
                  return (
                    <div
                      key={`${rowNumber}-${colNumber}`}
                      className="flex shrink-0 items-center border-r border-b border-[#2a2a2a] px-2"
                      style={{ width: COLUMN_WIDTH, height: ROW_HEIGHT }}
                      title={value}
                    >
                      <span className="truncate">{value}</span>
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
