import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

// Vitest config for the web suite. jsdom gives the components a DOM; the `@/*` alias mirrors
// tsconfig.json's paths so test imports resolve the same way the app does. The setup file installs
// Testing Library matchers + the EventSource mock (jsdom has no EventSource).
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./", import.meta.url)),
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./vitest.setup.ts"],
    include: ["**/*.test.ts", "**/*.test.tsx"],
    coverage: {
      provider: "v8",
      include: ["lib/**", "components/**"],
      reporter: ["text", "text-summary"],
    },
  },
});
