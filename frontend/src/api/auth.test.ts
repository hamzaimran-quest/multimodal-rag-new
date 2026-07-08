import { beforeEach, describe, expect, it, vi } from "vitest";

import { loginRequest, refreshRequest, signupRequest } from "./auth";

const okJson = (body: unknown) => ({ ok: true, json: async () => body });

describe("auth api", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("login posts credentials and includes cookies", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      okJson({ access_token: "tok", token_type: "bearer", user: { id: 1, email: "a@b.com" } }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await loginRequest("a@b.com", "supersecret1");

    expect(result.access_token).toBe("tok");
    expect(result.user.email).toBe("a@b.com");
    expect(fetchMock).toHaveBeenCalledWith(
      "/auth/login",
      expect.objectContaining({ method: "POST", credentials: "include" }),
    );
  });

  it("signup returns token payload", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        okJson({ access_token: "t2", token_type: "bearer", user: { id: 2, email: "c@d.com" } }),
      ),
    );

    const result = await signupRequest("c@d.com", "supersecret1");
    expect(result.user.id).toBe(2);
  });

  it("surfaces backend detail on error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        json: async () => ({ detail: "Invalid email or password" }),
      }),
    );

    await expect(loginRequest("a@b.com", "wrong")).rejects.toThrow("Invalid email or password");
  });

  it("extracts first validation message from array detail", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        json: async () => ({ detail: [{ msg: "value is not a valid email address" }] }),
      }),
    );

    await expect(signupRequest("bad", "supersecret1")).rejects.toThrow(
      "value is not a valid email address",
    );
  });

  it("refresh posts to refresh endpoint with cookies", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      okJson({ access_token: "r", token_type: "bearer", user: { id: 1, email: "a@b.com" } }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await refreshRequest();
    expect(fetchMock).toHaveBeenCalledWith(
      "/auth/refresh",
      expect.objectContaining({ method: "POST", credentials: "include" }),
    );
  });
});
