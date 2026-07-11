export interface HighlightRowRange {
  start: number;
  end: number;
}

export interface SpreadsheetHighlight {
  highlight: HighlightRowRange | null;
  scrollToRow: number | null;
}

const DEFAULT_MAX_HIGHLIGHT_SPAN = 24;

export interface SpreadsheetHighlightOptions {
  highlightRow?: number | null;
  maxHighlightSpan?: number;
}

/** Parse backend row_range / highlight_row into scroll + highlight targets. */
export function parseSpreadsheetHighlight(
  rowRange?: number[] | null,
  options?: SpreadsheetHighlightOptions,
): SpreadsheetHighlight {
  const maxHighlightSpan = options?.maxHighlightSpan ?? DEFAULT_MAX_HIGHLIGHT_SPAN;
  const highlightRow = options?.highlightRow;

  if (highlightRow != null && Number.isFinite(highlightRow) && highlightRow >= 1) {
    return {
      scrollToRow: highlightRow,
      highlight: { start: highlightRow, end: highlightRow },
    };
  }

  if (!rowRange || rowRange.length < 2) {
    return { highlight: null, scrollToRow: null };
  }
  const start = Math.min(rowRange[0], rowRange[1]);
  const end = Math.max(rowRange[0], rowRange[1]);
  if (!Number.isFinite(start) || !Number.isFinite(end) || start < 1) {
    return { highlight: null, scrollToRow: null };
  }
  const span = end - start + 1;
  return {
    scrollToRow: start,
    highlight: span <= maxHighlightSpan ? { start, end } : null,
  };
}

/** @deprecated Use parseSpreadsheetHighlight */
export function parseHighlightRowRange(rowRange?: number[] | null): HighlightRowRange | null {
  return parseSpreadsheetHighlight(rowRange).highlight;
}

/** Map a physical sheet row number to a grid row index. */
export function rowIndexForSheetRow(sheetRow: number, rowNumbers?: number[]): number {
  if (!Number.isFinite(sheetRow) || sheetRow < 1) {
    return 0;
  }
  if (rowNumbers?.length) {
    const exact = rowNumbers.indexOf(sheetRow);
    if (exact >= 0) {
      return exact;
    }
    const next = rowNumbers.findIndex((rowNumber) => rowNumber >= sheetRow);
    if (next >= 0) {
      return next;
    }
    return rowNumbers.length - 1;
  }
  return Math.max(0, sheetRow - 1);
}

export function isRowHighlighted(rowNumber: number, range: HighlightRowRange | null): boolean {
  if (!range) {
    return false;
  }
  return rowNumber >= range.start && rowNumber <= range.end;
}

/** Excel-style column label: 1 -> A, 26 -> Z, 27 -> AA. */
export function columnLabel(index: number): string {
  let remaining = index;
  let label = "";
  while (remaining > 0) {
    remaining -= 1;
    label = String.fromCharCode(65 + (remaining % 26)) + label;
    remaining = Math.floor(remaining / 26);
  }
  return label;
}
