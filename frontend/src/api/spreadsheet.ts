import { authFetch } from "./http";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await authFetch(path, init);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export interface SpreadsheetSheetMeta {
  name: string;
  index: number;
  row_count: number;
  col_count: number;
}

export interface SpreadsheetMetadata {
  doc_id: string;
  filename: string;
  sheet_count: number;
  sheets: SpreadsheetSheetMeta[];
}

export interface SpreadsheetSheetGrid {
  name: string;
  index: number;
  rows: string[][];
  row_count: number;
  col_count: number;
  highlight?: {
    row_start: number;
    row_end: number;
    col_start: number;
    col_end: number;
  } | null;
}

export async function getSpreadsheetMetadata(docId: string): Promise<SpreadsheetMetadata> {
  return request<SpreadsheetMetadata>(`/documents/${docId}/spreadsheet`);
}

export async function getSpreadsheetSheet(
  docId: string,
  sheetName: string,
  highlight?: {
    row_start?: number;
    row_end?: number;
    col_start?: number;
    col_end?: number;
  },
): Promise<SpreadsheetSheetGrid> {
  const params = new URLSearchParams();
  if (highlight?.row_start) params.set("row_start", String(highlight.row_start));
  if (highlight?.row_end) params.set("row_end", String(highlight.row_end));
  if (highlight?.col_start) params.set("col_start", String(highlight.col_start));
  if (highlight?.col_end) params.set("col_end", String(highlight.col_end));
  const query = params.toString();
  const suffix = query ? `?${query}` : "";
  return request<SpreadsheetSheetGrid>(`/documents/${docId}/spreadsheet/sheets/${encodeURIComponent(sheetName)}${suffix}`);
}
