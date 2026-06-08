import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

// Vitest config for the web suite. jsdom gives us a DOM for Testing Library; the @ alias mirrors
// tsconfig.json's paths ("@/*" -> "./*") so component/hook imports resolve identically to Next.
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
