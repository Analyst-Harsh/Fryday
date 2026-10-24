import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Dev: `make serve` runs the gateway on :8000 and this proxies the socket to it, so the
// page and /ws share an origin. Production: FastAPI serves dist/ itself (one deployable).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
      "/readyz": "http://127.0.0.1:8000",
    },
  },
  test: { environment: "node" },
});
