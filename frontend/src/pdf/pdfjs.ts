import * as pdfjsLib from "pdfjs-dist";
import type { PDFDocumentLoadingTask } from "pdfjs-dist";
// `?worker` lets Vite bundle the worker itself (works in dev + build), avoiding the
// "Setting up fake worker failed" error that `?url` module-import can trigger.
import PdfWorker from "pdfjs-dist/build/pdf.worker.min.mjs?worker";

import { getAccessToken } from "../auth/tokenStore";
import { pdfLog } from "./log";

try {
  pdfjsLib.GlobalWorkerOptions.workerPort = new PdfWorker();
  pdfLog("worker.init", { version: pdfjsLib.version, mode: "workerPort" });
} catch (err) {
  pdfLog("worker.init.error", { error: String(err) });
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

export function pdfFileUrl(docId: string): string {
  return `${API_BASE}/documents/${encodeURIComponent(docId)}/file`;
}

/**
 * Create a PDF.js loading task that streams via HTTP range requests.
 *
 * `disableAutoFetch` is the key flag: it stops PDF.js from eagerly pulling the
 * whole file after the first page, so opening a citation on page 340 never
 * downloads pages 1–339.
 */
export function createPdfLoadingTask(docId: string): PDFDocumentLoadingTask {
  const token = getAccessToken();
  return pdfjsLib.getDocument({
    url: pdfFileUrl(docId),
    httpHeaders: token ? { Authorization: `Bearer ${token}` } : undefined,
    withCredentials: true,
    disableAutoFetch: true,
    disableStream: false,
    rangeChunkSize: 65536,
  });
}

export { pdfjsLib };
