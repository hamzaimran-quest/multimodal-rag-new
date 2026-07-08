import { useCallback, useEffect, useRef, useState } from "react";

import { getDocumentStatus, isProcessing, listDocuments } from "../api/client";
import type { DocumentRecord } from "../types";

const POLL_MS = 2000;

export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const refresh = useCallback(async () => {
    try {
      setError(null);
      const docs = await listDocuments();
      setDocuments(docs);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load documents");
    } finally {
      setLoading(false);
    }
  }, []);

  const pollProcessing = useCallback(async () => {
    const processing = documents.filter((d) => isProcessing(d.ingestion_status));
    if (processing.length === 0) return;

    const updates = await Promise.all(
      processing.map((doc) => getDocumentStatus(doc.doc_id)),
    );

    setDocuments((current) => {
      const byId = new Map(updates.map((u) => [u.doc_id, u]));
      return current.map((doc) => byId.get(doc.doc_id) ?? doc);
    });
  }, [documents]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    const hasProcessing = documents.some((d) => isProcessing(d.ingestion_status));
    if (!hasProcessing) {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
      return;
    }

    if (!pollingRef.current) {
      pollingRef.current = setInterval(() => {
        void pollProcessing();
      }, POLL_MS);
    }

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [documents, pollProcessing]);

  return { documents, loading, error, refresh, setDocuments };
}
