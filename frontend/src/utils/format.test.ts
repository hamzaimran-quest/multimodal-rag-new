import {
  estimateIngestionProgress,
  formatUploadDate,
  statusClassName,
  statusLabel,
} from "../utils/format";

describe("format utils", () => {
  it("statusLabel maps known statuses", () => {
    expect(statusLabel("indexed")).toBe("Indexed");
    expect(statusLabel("processing")).toBe("Processing");
    expect(statusLabel("failed")).toBe("Failed");
  });

  it("statusClassName returns tailwind classes", () => {
    expect(statusClassName("indexed")).toContain("emerald");
    expect(statusClassName("failed")).toContain("rose");
  });

  it("formatUploadDate handles null", () => {
    expect(formatUploadDate(null)).toBe("—");
  });

  it("formatUploadDate formats iso strings", () => {
    const formatted = formatUploadDate("2025-06-01T12:00:00.000Z");
    expect(formatted).not.toBe("—");
  });

  it("estimateIngestionProgress returns full for terminal states", () => {
    expect(estimateIngestionProgress("indexed", null)).toBe(100);
    expect(estimateIngestionProgress("failed", null)).toBe(100);
  });
});
