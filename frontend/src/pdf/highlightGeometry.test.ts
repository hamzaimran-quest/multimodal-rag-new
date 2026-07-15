import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { bboxToNativePdfPoints } from "./highlightGeometry";

describe("bboxToNativePdfPoints", () => {
  it("passes a bbox through unchanged when the page's CropBox has a (0, 0) native origin", () => {
    // The common case (e.g. Huawei's report PDF): view = [0, 0, width, height].
    const pageView = [0, 0, 595.276, 841.89];
    const box: [number, number, number, number] = [70.866, 176.946, 538.0, 200.0];

    const native = bboxToNativePdfPoints(box, pageView);

    expect(native.x0).toBeCloseTo(70.866, 6);
    expect(native.y0).toBeCloseTo(841.89 - 176.946, 6);
    expect(native.x1).toBeCloseTo(538.0, 6);
    expect(native.y1).toBeCloseTo(841.89 - 200.0, 6);
  });

  it("re-anchors correctly when the page's CropBox has a non-zero native origin", () => {
    // Timberland's actual CropBox, expressed as pdf.js's native view: a page
    // whose visible region doesn't start at (0, 0) in the PDF's own coordinates
    // (e.g. print bleed/trim marks outside the CropBox). Naively passing (x0,
    // top) straight through -- the pre-fix bug -- would land tens of points off.
    const pageView = [0.0135956, 0.0135956, 593.986, 773.986];
    const box: [number, number, number, number] = [48.030609, 40.880153, 545.84622, 62.380093];

    const native = bboxToNativePdfPoints(box, pageView);

    // x: offset by the CropBox's native left edge (view[0]).
    expect(native.x0).toBeCloseTo(48.030609 + 0.0135956, 6);
    expect(native.x1).toBeCloseTo(545.84622 + 0.0135956, 6);
    // y: flipped via the CropBox's native top edge (view[3]) minus top/bottom.
    expect(native.y0).toBeCloseTo(773.986 - 40.880153, 6);
    expect(native.y1).toBeCloseTo(773.986 - 62.380093, 6);

    // The pre-fix formula used only the page's *height* (view[3] - view[1])
    // instead of view[3] itself, silently assuming view[1] === 0. Pin that this
    // would have been wrong here (a ~0.0136pt-scale check, but the same
    // omission at MediaBox scale is what caused the multi-line visible offset
    // reported against Timberland).
    const buggyHeight = pageView[3] - pageView[1];
    expect(buggyHeight - box[1]).not.toBeCloseTo(native.y0, 6);
  });
});

describe("bboxToNativePdfPoints against the real Timberland PDF (pdf.js, not mocked)", () => {
  it("places the page-15 'Loan Portfolio Analysis' heading near the top of the rendered page", async () => {
    const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");

    const fixturePath = path.resolve(
      path.dirname(fileURLToPath(import.meta.url)),
      "../../../test_data/timberland.pdf",
    );
    const data = new Uint8Array(await readFile(fixturePath));
    const doc = await pdfjs.getDocument({ data }).promise;

    try {
      const page = await doc.getPage(15);
      const viewport = page.getViewport({ scale: 1 });

      // bbox as actually emitted by the backend (CropBox-relative, see
      // app.ingestion.pdf_coords) for the chunk whose content is literally the
      // first heading on this page: "Loan Portfolio Analysis. The following
      // table sets forth the composition of the Bank's loan portfolio...".
      const box: [number, number, number, number] = [
        48.030609399999996, 40.880152655219945, 545.8462203999997, 62.38009265521998,
      ];

      const native = bboxToNativePdfPoints(box, page.view);
      const [, vy0] = viewport.convertToViewportPoint(native.x0, native.y0);
      const [, vy1] = viewport.convertToViewportPoint(native.x1, native.y1);

      const topFraction = Math.min(vy0, vy1) / viewport.height;
      const bottomFraction = Math.max(vy0, vy1) / viewport.height;

      // This is the page's first line of body text -- it must land near the
      // top (correct: ~0.053-0.081), not partway down the page. The pre-fix
      // combination (raw, unshifted backend bbox + the old formula that only
      // used the page's height instead of its CropBox top edge) lands this
      // same chunk at ~0.148 -- a full paragraph further down, on the wrong
      // text. 0.10 cleanly separates correct from buggy.
      expect(topFraction).toBeGreaterThan(0);
      expect(topFraction).toBeLessThan(0.1);
      expect(bottomFraction).toBeLessThan(0.1);
    } finally {
      await doc.destroy();
    }
  });
});
