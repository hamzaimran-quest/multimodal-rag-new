import { useState } from "react";

import { isProcessing } from "../api/client";
import { useDocuments } from "../hooks/useDocuments";
import type { DocumentRecord } from "../types";
import { DocumentTable } from "./DocumentTable";
import { UploadZone } from "./UploadZone";

export function SettingsTab() {
  const { documents, loading, error, refresh, setDocuments } = useDocuments();
  const [actionError, setActionError] = useState<string | null>(null);

  const processingCount = documents.filter((d) => isProcessing(d.ingestion_status)).length;

  const handleUploaded = (doc: DocumentRecord) => {
    setDocuments((current) => [doc, ...current.filter((d) => d.doc_id !== doc.doc_id)]);
    void refresh();
  };

  const handleDeleted = (docId: string) => {
    setDocuments((current) => current.filter((d) => d.doc_id !== docId));
  };

  const displayError = actionError ?? error;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-slate-100">Document library</h2>
        <p className="mt-1 text-sm text-slate-400">
          Upload PDFs for ingestion. Status updates automatically while processing.
        </p>
      </div>

      {processingCount > 0 && (
        <div className="flex items-center gap-3 rounded-lg border border-amber-500/25 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
          <span className="h-4 w-4 animate-spin rounded-full border-2 border-amber-300 border-t-transparent" />
          <span>
            {processingCount} document{processingCount > 1 ? "s are" : " is"} ingesting...
          </span>
        </div>
      )}

      <UploadZone onUploaded={handleUploaded} onError={(message) => setActionError(message)} />

      {displayError && (
        <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          {displayError}
        </div>
      )}

      {loading ? (
        <p className="text-sm text-slate-500">Loading documents…</p>
      ) : (
        <DocumentTable
          documents={documents}
          onDeleted={handleDeleted}
          onError={(message) => setActionError(message)}
        />
      )}
    </div>
  );
}
