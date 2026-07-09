import { deleteDocument } from "../api/client";
import type { DocumentRecord } from "../types";
import { formatUploadDate, statusClassName, statusLabel } from "../utils/format";
import { IngestionProgressRing } from "./IngestionProgressRing";

interface DocumentTableProps {
  documents: DocumentRecord[];
  onDeleted: (docId: string) => void;
  onError: (message: string) => void;
}

export function DocumentTable({ documents, onDeleted, onError }: DocumentTableProps) {
  const handleDelete = async (doc: DocumentRecord) => {
    try {
      await deleteDocument(doc.doc_id);
      onDeleted(doc.doc_id);
    } catch (err) {
      onError(err instanceof Error ? err.message : "Delete failed");
    }
  };

  if (documents.length === 0) {
    return (
      <p className="rounded-lg border border-slate-800 bg-slate-900/40 px-4 py-8 text-center text-sm text-slate-500">
        No documents uploaded yet.
      </p>
    );
  }

  return (
    <div className="overflow-hidden rounded-xl border border-slate-800">
      <table className="min-w-full divide-y divide-slate-800 text-sm">
        <thead className="bg-slate-900/80 text-left text-slate-400">
          <tr>
            <th className="px-4 py-3 font-medium">Filename</th>
            <th className="px-4 py-3 font-medium">Uploaded</th>
            <th className="px-4 py-3 font-medium">Progress</th>
            <th className="px-4 py-3 font-medium">Status</th>
            <th className="px-4 py-3 font-medium">Chunks</th>
            <th className="px-4 py-3 font-medium" />
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800 bg-slate-950/60">
          {documents.map((doc) => (
            <tr key={doc.doc_id} data-testid={`doc-row-${doc.doc_id}`}>
              <td className="px-4 py-3 font-medium text-slate-100">{doc.filename}</td>
              <td className="px-4 py-3 text-slate-400">{formatUploadDate(doc.upload_timestamp)}</td>
              <td className="px-4 py-3">
                <IngestionProgressRing
                  status={doc.ingestion_status}
                  uploadTimestamp={doc.upload_timestamp}
                  backendProgress={doc.ingestion_progress}
                  docId={doc.doc_id}
                />
              </td>
              <td className="px-4 py-3">
                <span
                  className={`inline-flex rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${statusClassName(doc.ingestion_status)}`}
                >
                  {statusLabel(doc.ingestion_status)}
                </span>
                {doc.error_message && (
                  <p className="mt-1 text-xs text-rose-400">{doc.error_message}</p>
                )}
                {doc.progress_message && doc.ingestion_status !== "failed" && (
                  <p className="mt-1 text-xs text-slate-400">{doc.progress_message}</p>
                )}
              </td>
              <td className="px-4 py-3 text-slate-300">{doc.chunk_count}</td>
              <td className="px-4 py-3 text-right">
                <button
                  type="button"
                  onClick={() => void handleDelete(doc)}
                  className="rounded-md px-2 py-1 text-xs text-rose-300 hover:bg-rose-500/10 hover:text-rose-200"
                >
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
