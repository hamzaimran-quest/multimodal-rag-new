import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { DocumentTable } from "../components/DocumentTable";
import type { DocumentRecord } from "../types";

const sampleDoc: DocumentRecord = {
  doc_id: "abc-123",
  filename: "huawei.pdf",
  ingestion_status: "indexed",
  ingestion_progress: 100,
  progress_message: "Completed",
  upload_timestamp: "2025-06-01T12:00:00.000Z",
  chunk_count: 128,
  error_message: null,
};

describe("DocumentTable", () => {
  it("renders empty state", () => {
    render(<DocumentTable documents={[]} onDeleted={() => {}} onError={() => {}} />);
    expect(screen.getByText(/no documents uploaded/i)).toBeInTheDocument();
  });

  it("renders document rows with status, chunk count, and progress ring", () => {
    render(<DocumentTable documents={[sampleDoc]} onDeleted={() => {}} onError={() => {}} />);

    expect(screen.getByText("huawei.pdf")).toBeInTheDocument();
    expect(screen.getByText("Indexed")).toBeInTheDocument();
    expect(screen.getByText("128")).toBeInTheDocument();
    expect(screen.getByTestId("doc-row-abc-123")).toBeInTheDocument();
    expect(screen.getByTestId("progress-ring-abc-123")).toBeInTheDocument();
  });

  it("calls delete immediately", async () => {
    const onDeleted = vi.fn();
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ doc_id: "abc-123", deleted_chunks: 10, status: "deleted" }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const user = (await import("@testing-library/user-event")).default.setup();
    render(
      <DocumentTable documents={[sampleDoc]} onDeleted={onDeleted} onError={() => {}} />,
    );

    await user.click(screen.getByRole("button", { name: /delete/i }));
    expect(onDeleted).toHaveBeenCalledWith("abc-123");
  });
});
