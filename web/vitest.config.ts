import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

// Vitest config for the web suite. jsdom gives the components a DOM; the @ alias mirrors
// tsconfig.json's paths ({ "@/*": ["./*"] }) so test imports resolve like the app does.
const root = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": root },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./vitest.setup.ts"],
    include: ["**/*.test.ts", "**/*.test.tsx"],
    coverage: {
      provider: "v8",
      reporter: ["text", "text-summary"],
      include: ["lib/**", "components/**"],
    },
  },
});
