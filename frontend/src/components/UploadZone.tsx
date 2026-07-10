import { useRef, useState } from "react";

import { uploadDocument } from "../api/client";
import type { DocumentRecord } from "../types";

interface UploadZoneProps {
  onUploaded: (doc: DocumentRecord) => void;
  onError: (message: string) => void;
}

export function UploadZone({ onUploaded, onError }: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);

  const handleFile = async (file: File | undefined) => {
    if (!file) return;
    if (!/\.(pdf|docx|xlsx)$/i.test(file.name)) {
      onError("Only PDF, DOCX, and XLSX files are supported.");
      return;
    }

    setUploading(true);
    try {
      const result = await uploadDocument(file);
      onUploaded({
        doc_id: result.doc_id,
        filename: result.filename,
        ingestion_status: "processing",
        ingestion_progress: 2,
        progress_message: "Queued",
        upload_timestamp: new Date().toISOString(),
        chunk_count: 0,
        error_message: null,
      });
    } catch (err) {
      onError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  return (
    <div className="rounded-xl border border-dashed border-slate-600 bg-slate-900/50 p-8 text-center">
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf,.pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,.docx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,.xlsx"
        className="hidden"
        data-testid="pdf-input"
        onChange={(e) => void handleFile(e.target.files?.[0])}
      />
      <p className="mb-4 text-sm text-slate-400">
        Upload a PDF, DOCX, or XLSX to index text and tables for retrieval.
      </p>
      <button
        type="button"
        disabled={uploading}
        onClick={() => inputRef.current?.click()}
        className="rounded-lg bg-indigo-500 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-60"
      >
        {uploading ? "Uploading…" : "Choose file"}
      </button>
    </div>
  );
}
