import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { PDFDocumentProxy, PDFPageProxy } from "pdfjs-dist";

import { getViewerConfig } from "../api/client";
import { pdfError, pdfLog } from "../pdf/log";
import { createPdfLoadingTask } from "../pdf/pdfjs";
import type { QuerySource } from "../types";

const PAGE_GAP = 16;
const DEFAULT_WINDOW = 2;
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 3;

export interface PdfViewerTarget {
  docId: string;
  filename: string;
  pageCount: number;
  sources: QuerySource[];
  chunkId: string;
  page: number;
}

interface PdfViewerPanelProps {
  target: PdfViewerTarget;
  onClose: () => void;
}

interface HighlightRect {
  key: string;
  chunkId: string;
  left: number;
  top: number;
  width: number;
  height: number;
  chunkType: string;
}

/** A chunk's highlight regions: its per-line boxes when available, else its union bbox. */
function highlightBoxesFor(source: QuerySource): number[][] {
  const lines = source.line_bboxes;
  if (Array.isArray(lines) && lines.length > 0) {
    return lines.filter((b) => Array.isArray(b) && b.length === 4);
  }
  if (Array.isArray(source.bbox) && source.bbox.length === 4) {
    return [source.bbox];
  }
  return [];
}

function viewerPageFor(source: QuerySource): number {
  if (source.source_format === "docx" && source.viewer_page) {
    return source.viewer_page;
  }
  return source.page_number;
}

function isHighlightable(source: QuerySource, docId: string): boolean {
  return source.doc_id === docId && highlightBoxesFor(source).length > 0;
}

export function PdfViewerPanel({ target, onClose }: PdfViewerPanelProps) {
  const { docId, filename, pageCount, sources, chunkId, page } = target;

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const pdfRef = useRef<PDFDocumentProxy | null>(null);

  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null);
  const [basePage, setBasePage] = useState<{ widthPts: number; heightPts: number } | null>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const [containerHeight, setContainerHeight] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [pageWindow, setPageWindow] = useState(DEFAULT_WINDOW);
  const [renderRange, setRenderRange] = useState<{ start: number; end: number }>({ start: 0, end: 0 });
  const [activeChunkId, setActiveChunkId] = useState(chunkId);

  const lastTargetKey = useRef<string>("");
  // Latest target page, read (not depended on) by the doc-load effect so that
  // retargeting to a new citation in the SAME document never reloads/destroys it.
  const pageRef = useRef(page);
  pageRef.current = page;

  // Fetch the configurable render-window size (falls back to default).
  useEffect(() => {
    let active = true;
    getViewerConfig()
      .then((cfg) => {
        if (active && typeof cfg.pdf_viewer_page_window === "number") {
          setPageWindow(Math.max(0, cfg.pdf_viewer_page_window));
        }
      })
      .catch(() => {
        /* keep default */
      });
    return () => {
      active = false;
    };
  }, []);

  // Load the PDF document (range-streamed) only when the DOCUMENT changes.
  // Retargeting to another citation in the same doc is handled by the scroll
  // effect below — it must not tear down and reload the live document.
  useEffect(() => {
    let cancelled = false;
    pdfLog("document.load.start", { docId });
    const task = createPdfLoadingTask(docId);
    setStatus("loading");
    setPdf(null);
    setBasePage(null);

    task.promise
      .then(async (doc) => {
        if (cancelled) return;
        pdfLog("document.load.ok", { docId, numPages: doc.numPages });
        pdfRef.current = doc;
        const firstPage = Math.min(Math.max(pageRef.current, 1), doc.numPages);
        const probe = await doc.getPage(firstPage);
        const viewport = probe.getViewport({ scale: 1 });
        if (cancelled) return;
        setPdf(doc);
        setBasePage({ widthPts: viewport.width, heightPts: viewport.height });
        setStatus("ready");
        pdfLog("document.ready", { docId, firstPage, widthPts: viewport.width, heightPts: viewport.height });
      })
      .catch((err) => {
        if (cancelled) return;
        pdfError("document.load.failed", err, { docId });
        setErrorMsg(err instanceof Error ? err.message : "Failed to load PDF");
        setStatus("error");
      });

    return () => {
      cancelled = true;
      task.destroy().catch(() => undefined);
      const doc = pdfRef.current;
      pdfRef.current = null;
      setPdf(null);
      if (doc) doc.destroy().catch(() => undefined);
    };
  }, [docId]);

  // Track container size for fit-to-width scaling and virtualization math.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width);
        setContainerHeight(entry.contentRect.height);
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [status]);

  const fitScale = basePage && containerWidth > 0 ? (containerWidth - 32) / basePage.widthPts : 0;
  const scale = fitScale > 0 ? fitScale * zoom : 0;
  const pageHeightPx = basePage ? basePage.heightPts * scale : 0;
  const pageWidthPx = basePage ? basePage.widthPts * scale : 0;
  const slotHeight = pageHeightPx + PAGE_GAP;
  const totalHeight = slotHeight > 0 ? slotHeight * pageCount : 0;

  const computeRange = useCallback(() => {
    const el = scrollRef.current;
    if (!el || slotHeight <= 0) return;
    const first = Math.floor(el.scrollTop / slotHeight);
    const last = Math.floor((el.scrollTop + el.clientHeight) / slotHeight);
    const start = Math.max(0, first - pageWindow);
    const end = Math.min(pageCount - 1, last + pageWindow);
    setRenderRange((prev) => {
      if (prev.start === start && prev.end === end) return prev;
      pdfLog("render.window", { start: start + 1, end: end + 1, pageWindow, rendered: end - start + 1 });
      return { start, end };
    });
  }, [slotHeight, pageWindow, pageCount]);

  const handleScroll = useCallback(() => computeRange(), [computeRange]);

  // Recompute the render window whenever geometry changes.
  useEffect(() => {
    computeRange();
  }, [computeRange, scale, containerHeight]);

  // Center the cited page on open and on retarget (new clicked source).
  useLayoutEffect(() => {
    if (status !== "ready" || slotHeight <= 0) return;
    const key = `${docId}:${chunkId}:${page}`;
    if (lastTargetKey.current === key) return;
    lastTargetKey.current = key;
    setActiveChunkId(chunkId);
    const el = scrollRef.current;
    if (!el) return;
    const target = (page - 1) * slotHeight - Math.max(0, (el.clientHeight - pageHeightPx) / 2);
    el.scrollTop = Math.max(0, Math.min(target, totalHeight - el.clientHeight));
    computeRange();
  }, [status, slotHeight, docId, chunkId, page, pageHeightPx, totalHeight, computeRange]);

  const highlightsByPage = useMemo(() => {
    const map = new Map<number, QuerySource[]>();
    for (const source of sources) {
      if (!isHighlightable(source, docId)) continue;
      const page = viewerPageFor(source);
      const list = map.get(page) ?? [];
      list.push(source);
      map.set(page, list);
    }
    return map;
  }, [sources, docId]);

  const selectChunk = useCallback(
    (selectedChunkId: string, pageNumber: number) => {
      setActiveChunkId(selectedChunkId);
      const el = scrollRef.current;
      if (!el || slotHeight <= 0) return;
      const target = (pageNumber - 1) * slotHeight - Math.max(0, (el.clientHeight - pageHeightPx) / 2);
      el.scrollTo({ top: Math.max(0, Math.min(target, totalHeight - el.clientHeight)), behavior: "smooth" });
    },
    [slotHeight, pageHeightPx, totalHeight],
  );

  const renderedPages: number[] = [];
  if (status === "ready" && slotHeight > 0) {
    for (let i = renderRange.start; i <= renderRange.end; i++) renderedPages.push(i);
  }

  return (
    <div className="fixed right-0 top-0 z-40 flex h-screen w-[min(52vw,760px)] flex-col border-l border-[#2a2a2a] bg-[#0f0f0f] shadow-2xl shadow-black/60 max-[880px]:w-full">
      <div className="flex shrink-0 items-center justify-between gap-3 border-b border-[#2a2a2a] bg-[#141414] px-4 py-3">
        <div className="min-w-0">
          <div className="truncate text-[13.5px] font-semibold text-[#f5f5f5]">{filename}</div>
          <div className="text-[11px] text-[#737373]">
            {pageCount > 0 ? `${pageCount} pages` : "PDF"} · viewing citation
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => setZoom((z) => Math.max(MIN_ZOOM, Math.round((z - 0.25) * 100) / 100))}
            className="h-7 w-7 rounded-[6px] border border-[#333333] text-[#a3a3a3] hover:border-[#525252] hover:text-[#e5e5e5]"
            aria-label="Zoom out"
          >
            −
          </button>
          <button
            type="button"
            onClick={() => setZoom(1)}
            className="h-7 rounded-[6px] border border-[#333333] px-2 text-[11px] text-[#a3a3a3] hover:border-[#525252] hover:text-[#e5e5e5]"
            aria-label="Fit to width"
          >
            {Math.round(zoom * 100)}%
          </button>
          <button
            type="button"
            onClick={() => setZoom((z) => Math.min(MAX_ZOOM, Math.round((z + 0.25) * 100) / 100))}
            className="h-7 w-7 rounded-[6px] border border-[#333333] text-[#a3a3a3] hover:border-[#525252] hover:text-[#e5e5e5]"
            aria-label="Zoom in"
          >
            +
          </button>
          <button
            type="button"
            onClick={onClose}
            className="ml-1 h-7 w-7 rounded-[6px] border border-[#333333] text-[#a3a3a3] hover:border-[#525252] hover:text-[#e5e5e5]"
            aria-label="Close viewer"
          >
            ✕
          </button>
        </div>
      </div>

      <div ref={scrollRef} onScroll={handleScroll} className="relative flex-1 overflow-y-auto bg-[#0a0a0a]">
        {status === "loading" && (
          <div className="flex h-full items-center justify-center text-[13px] text-[#737373]">Loading document…</div>
        )}
        {status === "error" && (
          <div className="flex h-full items-center justify-center px-6 text-center text-[13px] text-rose-300">
            {errorMsg || "Failed to load PDF"}
          </div>
        )}
        {status === "ready" && pdf && basePage && (
          <div style={{ height: totalHeight, position: "relative" }}>
            {renderedPages.map((index) => (
              <div
                key={index}
                style={{
                  position: "absolute",
                  top: index * slotHeight,
                  left: "50%",
                  transform: "translateX(-50%)",
                  width: pageWidthPx,
                  height: pageHeightPx,
                }}
              >
                <PdfPage
                  pdf={pdf}
                  pageNumber={index + 1}
                  scale={scale}
                  width={pageWidthPx}
                  height={pageHeightPx}
                  sources={highlightsByPage.get(index + 1) ?? []}
                  activeChunkId={activeChunkId}
                  onSelect={(id) => selectChunk(id, index + 1)}
                />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

interface PdfPageProps {
  pdf: PDFDocumentProxy;
  pageNumber: number;
  scale: number;
  width: number;
  height: number;
  sources: QuerySource[];
  activeChunkId: string | null;
  onSelect: (chunkId: string) => void;
}

function PdfPage({ pdf, pageNumber, scale, width, height, sources, activeChunkId, onSelect }: PdfPageProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [rendered, setRendered] = useState(false);
  const [rects, setRects] = useState<HighlightRect[]>([]);

  // Canvas rendering — depends only on page + scale, never on which citation is
  // active, so retargeting a citation does not re-rasterize the page.
  useEffect(() => {
    let cancelled = false;
    let renderTask: ReturnType<PDFPageProxy["render"]> | null = null;
    setRendered(false);

    pdf
      .getPage(pageNumber)
      .then((pdfPage) => {
        if (cancelled) return;
        const viewport = pdfPage.getViewport({ scale });
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        canvas.width = Math.floor(viewport.width);
        canvas.height = Math.floor(viewport.height);

        renderTask = pdfPage.render({ canvasContext: ctx, viewport });
        renderTask.promise
          .then(() => {
            if (!cancelled) {
              setRendered(true);
              pdfLog("page.render.ok", { pageNumber, scale: Math.round(scale * 100) / 100 });
            }
          })
          .catch((err) => {
            // RenderingCancelledException is expected on unmount/scale change; ignore it.
            if (!cancelled && !(err && err.name === "RenderingCancelledException")) {
              pdfError("page.render.failed", err, { pageNumber });
            }
          });
      })
      .catch((err) => {
        if (!cancelled) pdfError("page.get.failed", err, { pageNumber });
      });

    return () => {
      cancelled = true;
      if (renderTask) renderTask.cancel();
    };
  }, [pdf, pageNumber, scale]);

  // Highlight geometry — recomputed from stored PDF-space bboxes, scaled to the
  // current viewport so boxes stay correct at any zoom/panel width.
  useEffect(() => {
    let cancelled = false;
    pdf
      .getPage(pageNumber)
      .then((pdfPage) => {
        if (cancelled) return;
        const viewport = pdfPage.getViewport({ scale });
        const pageHeightPts = pdfPage.view[3] - pdfPage.view[1];
        const computed: HighlightRect[] = [];
        for (const source of sources) {
          highlightBoxesFor(source).forEach((box, i) => {
            const [x0, top, x1, bottom] = box;
            const [vx0, vy0] = viewport.convertToViewportPoint(x0, pageHeightPts - top);
            const [vx1, vy1] = viewport.convertToViewportPoint(x1, pageHeightPts - bottom);
            computed.push({
              key: `${source.chunk_id}-${i}`,
              chunkId: source.chunk_id,
              left: Math.min(vx0, vx1),
              top: Math.min(vy0, vy1),
              width: Math.abs(vx1 - vx0),
              height: Math.abs(vy1 - vy0),
              chunkType: source.chunk_type,
            });
          });
        }
        if (!cancelled) setRects(computed);
      })
      .catch(() => {
        /* page unavailable (doc switching); highlights will recompute on next load */
      });
    return () => {
      cancelled = true;
    };
  }, [pdf, pageNumber, scale, sources]);

  return (
    <div className="relative h-full w-full">
      {!rendered && (
        <div
          className="absolute inset-0 flex items-center justify-center rounded-[2px] border border-[#1f1f1f] bg-[#141414] text-[11px] text-[#525252]"
          style={{ width, height }}
        >
          Page {pageNumber}
        </div>
      )}
      <canvas ref={canvasRef} className="block bg-white" style={{ width, height }} />
      {rendered &&
        rects.map((rect) => {
          const isPrimary = rect.chunkId === activeChunkId;
          return (
            <button
              key={rect.key}
              type="button"
              onClick={() => onSelect(rect.chunkId)}
              title={`Citation on page ${pageNumber}`}
              className={`absolute cursor-pointer transition-colors ${
                isPrimary
                  ? "bg-amber-300/25 hover:bg-amber-300/35"
                  : "bg-sky-300/15 hover:bg-sky-300/25"
              }`}
              style={{
                left: rect.left,
                top: rect.top,
                width: rect.width,
                height: rect.height,
                zIndex: isPrimary ? 2 : 1,
              }}
            />
          );
        })}
    </div>
  );
}
