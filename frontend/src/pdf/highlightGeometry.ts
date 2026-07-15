/**
 * Converts a stored citation bbox (backend PDF-space) into the native PDF point
 * pair that pdf.js's `PageViewport.convertToViewportPoint` expects.
 *
 * The backend (app.ingestion.pdf_coords) stores bbox/line_bboxes relative to the
 * page's own CropBox — the region a PDF renderer actually displays — using
 * pdfplumber's top-down convention (x0 from the left edge, top/bottom as
 * distance down from the CropBox's top edge). pdf.js's `PDFPageProxy.view` is
 * that same CropBox expressed as native, bottom-up PDF coordinates:
 * `[x0, y0, x1, y1]`. Re-anchoring requires both:
 *   - x: offset by view[0] (the CropBox's native left edge)
 *   - y: flipped via view[3] - top (the CropBox's native top edge, since
 *     "top" measures downward from it)
 *
 * Most PDFs have a CropBox at (0, 0), which is why a naive pass-through of
 * (x0, top) looks correct there — it only breaks on PDFs whose CropBox has a
 * non-zero native origin (e.g. print bleed/trim marks pushing the visible page
 * away from the PDF's coordinate origin).
 */
export function bboxToNativePdfPoints(
  box: readonly [number, number, number, number],
  pageView: readonly number[],
): { x0: number; y0: number; x1: number; y1: number } {
  const [pageX0, , , pageY1] = pageView;
  const [x0, top, x1, bottom] = box;
  return {
    x0: pageX0 + x0,
    y0: pageY1 - top,
    x1: pageX0 + x1,
    y1: pageY1 - bottom,
  };
}
