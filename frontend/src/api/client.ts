import type {
  DeleteDocumentResponse,
  DocumentListResponse,
  DocumentRecord,
  UploadResponse,
} from "../types";
import { authFetch } from "./http";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await authFetch(path, init);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function listDocuments(): Promise<DocumentRecord[]> {
  const data = await request<DocumentListResponse>("/documents");
  return data.documents;
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  return request<UploadResponse>("/documents/upload", {
    method: "POST",
    body: form,
  });
}

export async function getDocumentStatus(docId: string): Promise<DocumentRecord> {
  return request<DocumentRecord>(`/documents/${docId}/status`);
}

export async function deleteDocument(docId: string): Promise<DeleteDocumentResponse> {
  return request<DeleteDocumentResponse>(`/documents/${docId}`, {
    method: "DELETE",
  });
}

export function isProcessing(status: string): boolean {
  return status === "pending" || status === "processing";
}

export async function getViewerConfig(): Promise<{ pdf_viewer_page_window: number }> {
  return request<{ pdf_viewer_page_window: number }>("/config");
}

export interface EmbeddingModelStatus {
  state: "not_started" | "loading" | "ready" | "error";
  started_at: number | null;
  finished_at: number | null;
  error: string | null;
}

export async function getEmbeddingModelStatus(): Promise<EmbeddingModelStatus> {
  return request<EmbeddingModelStatus>("/embedding-model-status");
}
