import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { FlatCompat } from "@eslint/eslintrc";

// eslint-config-next is still authored in the legacy (eslintrc) format, so we bridge it into
// flat config with FlatCompat. This is the route Next 15 itself scaffolds for ESLint 9 flat
// config. `next/core-web-vitals` pulls in next, react, react-hooks, and jsx-a11y rules;
// `next/typescript` layers on @typescript-eslint for the TS sources.
const compat = new FlatCompat({ baseDirectory: dirname(fileURLToPath(import.meta.url)) });

const eslintConfig = [
  // Ignores live in the flat config (no separate .eslintignore in ESLint 9). Build output,
  // deps, and coverage artifacts are never linted.
  {
    ignores: [".next/**", "out/**", "node_modules/**", "coverage/**", "next-env.d.ts"],
  },
  ...compat.extends("next/core-web-vitals", "next/typescript"),
];

export default eslintConfig;
