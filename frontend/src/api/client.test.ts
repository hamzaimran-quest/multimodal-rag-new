import { describe, expect, it, vi, beforeEach } from "vitest";

import {
  deleteDocument,
  getDocumentStatus,
  isProcessing,
  listDocuments,
  uploadDocument,
} from "../api/client";

const authFetchMock = vi.fn();

vi.mock("./http", () => ({
  authFetch: (...args: unknown[]) => authFetchMock(...args),
}));

describe("api client", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("listDocuments fetches document list", async () => {
    authFetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        documents: [
          {
            doc_id: "d1",
            filename: "huawei.pdf",
            ingestion_status: "indexed",
            ingestion_progress: 100,
            progress_message: "Completed",
            upload_timestamp: "2025-01-01T00:00:00Z",
            chunk_count: 42,
            error_message: null,
          },
        ],
      }),
    });

    const docs = await listDocuments();
    expect(docs).toHaveLength(1);
    expect(docs[0].filename).toBe("huawei.pdf");
    expect(authFetchMock).toHaveBeenCalledWith("/documents", undefined);
  });

  it("uploadDocument posts multipart form", async () => {
    authFetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        doc_id: "new-id",
        filename: "test.pdf",
        status: "processing",
      }),
    });

    const file = new File(["pdf"], "test.pdf", { type: "application/pdf" });
    const result = await uploadDocument(file);

    expect(result.doc_id).toBe("new-id");
    expect(authFetchMock).toHaveBeenCalledWith(
      "/documents/upload",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("getDocumentStatus fetches status endpoint", async () => {
    authFetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        doc_id: "d1",
        filename: "a.pdf",
        ingestion_status: "processing",
        ingestion_progress: 42,
        progress_message: "Parsed page 3/7",
        upload_timestamp: null,
        chunk_count: 0,
        error_message: null,
      }),
    });

    const status = await getDocumentStatus("d1");
    expect(status.ingestion_status).toBe("processing");
  });

  it("deleteDocument sends DELETE", async () => {
    authFetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ doc_id: "d1", deleted_chunks: 5, status: "deleted" }),
    });

    const result = await deleteDocument("d1");
    expect(result.deleted_chunks).toBe(5);
    expect(authFetchMock).toHaveBeenCalledWith(
      "/documents/d1",
      expect.objectContaining({ method: "DELETE" }),
    );
  });

  it("isProcessing detects active statuses", () => {
    expect(isProcessing("processing")).toBe(true);
    expect(isProcessing("pending")).toBe(true);
    expect(isProcessing("indexed")).toBe(false);
    expect(isProcessing("failed")).toBe(false);
  });
});
