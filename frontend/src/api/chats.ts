import type { ComputedChart, QuerySource, SqlMeta } from "../types";
import { authFetch } from "./http";

export interface ChatSessionSummary {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ChatMessageRecord {
  id: number;
  role: "user" | "assistant";
  content: string;
  sources: QuerySource[];
  charts: ComputedChart[];
  sql_meta?: SqlMeta | null;
  created_at: string;
}

export interface ChatSessionDetail extends ChatSessionSummary {
  messages: ChatMessageRecord[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await authFetch(path, init);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function listChats(): Promise<ChatSessionSummary[]> {
  const data = await request<{ sessions: ChatSessionSummary[] }>("/chats");
  return data.sessions;
}

export async function createChat(): Promise<ChatSessionSummary> {
  return request<ChatSessionSummary>("/chats", { method: "POST" });
}

export async function getChat(sessionId: number): Promise<ChatSessionDetail> {
  return request<ChatSessionDetail>(`/chats/${sessionId}`);
}

export async function deleteChat(sessionId: number): Promise<void> {
  await request(`/chats/${sessionId}`, { method: "DELETE" });
}
