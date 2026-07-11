import { describe, expect, it } from "vitest";

import { columnLabel } from "./spreadsheet";

describe("columnLabel", () => {
  it("maps indices to Excel-style labels", () => {
    expect(columnLabel(1)).toBe("A");
    expect(columnLabel(26)).toBe("Z");
    expect(columnLabel(27)).toBe("AA");
  });
});
