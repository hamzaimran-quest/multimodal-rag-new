import { describe, expect, it } from "vitest";

import { cellHighlightClass, columnLabel } from "./spreadsheet";

describe("columnLabel", () => {
  it("maps indices to Excel-style labels", () => {
    expect(columnLabel(1)).toBe("A");
    expect(columnLabel(26)).toBe("Z");
    expect(columnLabel(27)).toBe("AA");
  });
});

describe("cellHighlightClass", () => {
  it("highlights cells inside the cited range", () => {
    expect(cellHighlightClass(5, 2, [4, 6], [1, 3])).toContain("bg-amber-500/20");
    expect(cellHighlightClass(2, 2, [4, 6], [1, 3])).toBe("");
  });
});
