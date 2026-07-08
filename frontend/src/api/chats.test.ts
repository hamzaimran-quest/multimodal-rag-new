import { beforeEach, describe, expect, it, vi } from "vitest";

import { createChat, deleteChat, getChat, listChats } from "./chats";

const authFetchMock = vi.fn();

vi.mock("./http", () => ({
  authFetch: (...args: unknown[]) => authFetchMock(...args),
}));

describe("chats api", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("listChats fetches sessions", async () => {
    authFetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        sessions: [{ id: 1, title: "Revenue question", created_at: "t", updated_at: "t" }],
      }),
    });

    const sessions = await listChats();
    expect(sessions).toHaveLength(1);
    expect(authFetchMock).toHaveBeenCalledWith("/chats", undefined);
  });

  it("createChat posts new session", async () => {
    authFetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ id: 2, title: "New chat", created_at: "t", updated_at: "t" }),
    });

    const session = await createChat();
    expect(session.id).toBe(2);
    expect(authFetchMock).toHaveBeenCalledWith("/chats", { method: "POST" });
  });

  it("getChat loads messages", async () => {
    authFetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        id: 3,
        title: "Chat",
        created_at: "t",
        updated_at: "t",
        messages: [{ id: 10, role: "user", content: "hi", sources: [], charts: [], created_at: "t" }],
      }),
    });

    const detail = await getChat(3);
    expect(detail.messages[0].content).toBe("hi");
    expect(authFetchMock).toHaveBeenCalledWith("/chats/3", undefined);
  });

  it("deleteChat sends DELETE", async () => {
    authFetchMock.mockResolvedValue({ ok: true, json: async () => ({ session_id: 4, status: "deleted" }) });

    await deleteChat(4);
    expect(authFetchMock).toHaveBeenCalledWith("/chats/4", { method: "DELETE" });
  });
});
