import type { SqlAgentStatus, SqlConnection } from "../types";
import { authFetch } from "./http";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await authFetch(path, init);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export async function getSqlAgentStatus(): Promise<SqlAgentStatus> {
  return request<SqlAgentStatus>("/sql-agent/status");
}

export async function addSqlConnection(payload: {
  connection_url: string;
  display_name: string;
  description: string;
  activate?: boolean;
}): Promise<SqlConnection> {
  return request<SqlConnection>("/sql-agent/connections", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function activateSqlConnection(connectionId: number): Promise<SqlConnection> {
  return request<SqlConnection>(`/sql-agent/connections/${connectionId}/activate`, { method: "POST" });
}

export async function testSqlConnection(connectionId: number): Promise<SqlConnection> {
  return request<SqlConnection>(`/sql-agent/connections/${connectionId}/test`, { method: "POST" });
}

export async function updateSqlConnection(
  connectionId: number,
  payload: { display_name?: string; description?: string },
): Promise<SqlConnection> {
  return request<SqlConnection>(`/sql-agent/connections/${connectionId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function updateSqlCredentials(connectionId: number, connection_url: string): Promise<SqlConnection> {
  return request<SqlConnection>(`/sql-agent/connections/${connectionId}/credentials`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ connection_url }),
  });
}

export async function deleteSqlConnection(connectionId: number): Promise<void> {
  await request<void>(`/sql-agent/connections/${connectionId}`, { method: "DELETE" });
}

export async function deactivateSqlConnections(): Promise<void> {
  await request<void>("/sql-agent/deactivate", { method: "POST" });
}
