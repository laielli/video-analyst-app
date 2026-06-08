import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";

// Vitest config for the web test suite (the project's first JS tests).
// - jsdom environment: components render under React Testing Library; useRun.ts touches the
//   browser EventSource API (mocked in vitest.setup.ts) which jsdom does not implement.
// - @/* alias mirrors tsconfig.json's paths so component imports (`@/lib/types`) resolve.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": resolve(__dirname, ".") },
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
