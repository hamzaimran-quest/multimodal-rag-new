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
  row_numbers?: number[];
  row_count: number;
  col_count: number;
}

export async function getSpreadsheetMetadata(docId: string): Promise<SpreadsheetMetadata> {
  return request<SpreadsheetMetadata>(`/documents/${docId}/spreadsheet`);
}

export async function getSpreadsheetSheet(
  docId: string,
  sheetName: string,
): Promise<SpreadsheetSheetGrid> {
  return request<SpreadsheetSheetGrid>(
    `/documents/${docId}/spreadsheet/sheets/${encodeURIComponent(sheetName)}`,
  );
}
