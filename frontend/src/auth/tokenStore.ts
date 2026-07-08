// In-memory access token store. Kept out of localStorage to reduce XSS exposure;
// the httpOnly refresh cookie restores the session on reload.
let accessToken: string | null = null;

export function getAccessToken(): string | null {
  return accessToken;
}

export function setAccessToken(token: string | null): void {
  accessToken = token;
}
