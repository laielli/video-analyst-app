import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

// Mirror tsconfig's `@/*` -> `./*` path alias so component/lib imports resolve in tests.
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
      // Calibrated floors: ~5pp below the natural numbers (lines/statements 99.03%,
      // functions 100%, branches 91.9%), floored to a multiple of 5. `npm run test:cov`
      // (and the web CI job) fails if coverage drops below these.
      thresholds: {
        lines: 95,
        statements: 95,
        functions: 95,
        branches: 85,
      },
    },
  },
});
