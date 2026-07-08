export type IngestionStatus = "pending" | "processing" | "indexed" | "failed";

export interface DocumentRecord {
  doc_id: string;
  filename: string;
  ingestion_status: IngestionStatus;
  ingestion_progress: number;
  progress_message: string | null;
  upload_timestamp: string | null;
  chunk_count: number;
  error_message: string | null;
}

export interface DocumentListResponse {
  documents: DocumentRecord[];
}

export interface UploadResponse {
  doc_id: string;
  filename: string;
  status: string;
}

export interface DeleteDocumentResponse {
  doc_id: string;
  deleted_chunks: number;
  status: string;
}

export interface AttachedImage {
  image_chunk_id: string;
  doc_id?: string | null;
  filename?: string | null;
  page_number?: number | null;
  image_url: string;
  bbox?: number[] | null;
  caption?: string;
  score: number;
  reason: "intent" | "proximity" | string;
}

export interface QuerySource {
  chunk_id: string;
  doc_id?: string;
  filename: string;
  page_number: number;
  chunk_type: "text" | "table" | "image" | string;
  snippet: string;
  image_url?: string | null;
  score: number;
  source_format?: "pdf" | "docx" | string;
  section?: string | null;
  viewer_page?: number | null;
  bbox?: number[] | null;
  line_bboxes?: number[][] | null;
  page_count?: number;
  attached_images?: AttachedImage[] | null;
  attach_reason?: "intent" | string | null;
}

export type { ComputedChart, ChartSeries } from "./types/charts";
