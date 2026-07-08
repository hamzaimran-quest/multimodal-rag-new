import { refreshRequest } from "./auth";
import { getAccessToken, setAccessToken } from "../auth/tokenStore";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

/**
 * fetch wrapper that attaches the in-memory access token and, on a 401,
 * transparently refreshes once (via the httpOnly cookie) and retries.
 */
export async function authFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const call = (token: string | null): Promise<Response> => {
    const headers = new Headers(init.headers);
    if (token) headers.set("Authorization", `Bearer ${token}`);
    return fetch(`${API_BASE}${path}`, { ...init, headers, credentials: "include" });
  };

  let response = await call(getAccessToken());
  if (response.status !== 401) return response;

  try {
    const refreshed = await refreshRequest();
    setAccessToken(refreshed.access_token);
    response = await call(refreshed.access_token);
  } catch {
    setAccessToken(null);
  }
  return response;
}
