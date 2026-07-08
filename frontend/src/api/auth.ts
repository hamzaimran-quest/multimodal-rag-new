const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

export interface AuthUser {
  id: number;
  email: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: AuthUser;
}

function extractErrorMessage(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length > 0) {
    const first = detail[0] as { msg?: string };
    if (first?.msg) return first.msg;
  }
  return fallback;
}

async function parseError(response: Response, fallback: string): Promise<Error> {
  try {
    const body = await response.json();
    return new Error(extractErrorMessage(body?.detail, fallback));
  } catch {
    return new Error(fallback);
  }
}

export async function signupRequest(email: string, password: string): Promise<TokenResponse> {
  const response = await fetch(`${API_BASE}/auth/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ email, password }),
  });
  if (!response.ok) throw await parseError(response, "Signup failed");
  return response.json() as Promise<TokenResponse>;
}

export async function loginRequest(email: string, password: string): Promise<TokenResponse> {
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ email, password }),
  });
  if (!response.ok) throw await parseError(response, "Invalid email or password");
  return response.json() as Promise<TokenResponse>;
}

export async function refreshRequest(): Promise<TokenResponse> {
  const response = await fetch(`${API_BASE}/auth/refresh`, {
    method: "POST",
    credentials: "include",
  });
  if (!response.ok) throw await parseError(response, "Session expired");
  return response.json() as Promise<TokenResponse>;
}

export async function logoutRequest(): Promise<void> {
  await fetch(`${API_BASE}/auth/logout`, {
    method: "POST",
    credentials: "include",
  });
}
