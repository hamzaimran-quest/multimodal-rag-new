/** Client-side diagnostics for spreadsheet row highlighting. */
export function logSpreadsheetHighlight(
  event: string,
  payload: Record<string, unknown>,
): void {
  console.info(`[spreadsheet-highlight] ${event}`, payload);
}
