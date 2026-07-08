/**
 * Lightweight, prefixed logger for the PDF viewer so worker/load/render failures
 * are easy to spot and copy from the browser console.
 *
 * Enabled by default in dev; in production set `localStorage.pdfDebug = "1"` to turn on.
 */
const ENABLED =
  import.meta.env.DEV ||
  (typeof localStorage !== "undefined" && localStorage.getItem("pdfDebug") === "1");

const PREFIX = "[pdf-viewer]";

export function pdfLog(event: string, detail?: Record<string, unknown>): void {
  if (!ENABLED) return;
  if (detail) {
    console.info(`${PREFIX} ${event}`, detail);
  } else {
    console.info(`${PREFIX} ${event}`);
  }
}

export function pdfError(event: string, error: unknown, detail?: Record<string, unknown>): void {
  // Errors always log, regardless of debug flag.
  console.error(`${PREFIX} ${event}`, { error: error instanceof Error ? error.message : String(error), ...detail });
}
