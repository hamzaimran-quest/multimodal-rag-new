import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiProxyTarget = process.env.VITE_API_PROXY_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  // Same-origin static assets for FastAPI (/app/static in Docker).
  build: {
    outDir: "../backend/static",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/auth": apiProxyTarget,
      "/documents": apiProxyTarget,
      "/search": apiProxyTarget,
      "/query": apiProxyTarget,
      "/chats": apiProxyTarget,
      "/health": apiProxyTarget,
      "/ready": apiProxyTarget,
      "/config": apiProxyTarget,
      "/images": apiProxyTarget,
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    globals: true,
  },
});
