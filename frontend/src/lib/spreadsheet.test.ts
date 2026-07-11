import { describe, expect, it } from "vitest";

import { columnLabel, isRowHighlighted, parseHighlightRowRange, parseSpreadsheetHighlight, rowIndexForSheetRow } from "./spreadsheet";

describe("columnLabel", () => {
  it("maps indices to Excel-style labels", () => {
    expect(columnLabel(1)).toBe("A");
    expect(columnLabel(26)).toBe("Z");
    expect(columnLabel(27)).toBe("AA");
  });
});

describe("parseSpreadsheetHighlight", () => {
  it("prefers explicit highlight_row", () => {
    expect(parseSpreadsheetHighlight([2, 500], { highlightRow: 42 })).toEqual({
      scrollToRow: 42,
      highlight: { start: 42, end: 42 },
    });
  });

  it("highlights narrow ranges and always scrolls", () => {
    expect(parseSpreadsheetHighlight([42, 42])).toEqual({
      scrollToRow: 42,
      highlight: { start: 42, end: 42 },
    });
  });

  it("scrolls without highlighting wide ranges", () => {
    expect(parseSpreadsheetHighlight([2, 500])).toEqual({
      scrollToRow: 2,
      highlight: null,
    });
  });
});

describe("parseHighlightRowRange", () => {
  it("normalizes backend row ranges", () => {
    expect(parseHighlightRowRange([42, 42])).toEqual({ start: 42, end: 42 });
    expect(parseHighlightRowRange([8, 4])).toEqual({ start: 4, end: 8 });
  });

  it("returns null for invalid ranges", () => {
    expect(parseHighlightRowRange(null)).toBeNull();
    expect(parseHighlightRowRange([0, 5])).toBeNull();
  });
});

describe("rowIndexForSheetRow", () => {
  it("maps physical sheet rows using row_numbers", () => {
    expect(rowIndexForSheetRow(42, [1, 2, 40, 41, 42])).toBe(4);
    expect(rowIndexForSheetRow(39, [1, 2, 40, 41, 42])).toBe(2);
  });

  it("falls back to contiguous numbering", () => {
    expect(rowIndexForSheetRow(42)).toBe(41);
  });
});

describe("isRowHighlighted", () => {
  it("checks inclusive row numbers", () => {
    const range = { start: 4, end: 6 };
    expect(isRowHighlighted(3, range)).toBe(false);
    expect(isRowHighlighted(4, range)).toBe(true);
    expect(isRowHighlighted(6, range)).toBe(true);
    expect(isRowHighlighted(7, range)).toBe(false);
  });
});
