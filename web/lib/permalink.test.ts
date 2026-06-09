import { describe, it, expect } from "vitest";
import { permalinkFor } from "@/lib/useRun";

// The shareable-link helper is a pure function (no clipboard, no window) so the URL shape can be
// pinned in isolation: ${origin}/?run=<run_id>, with the id percent-encoded.

describe("permalinkFor", () => {
  it("builds ${origin}/?run=<run_id>", () => {
    expect(permalinkFor("https://demo.example", "abc123")).toBe("https://demo.example/?run=abc123");
  });

  it("percent-encodes the run id (defense in depth on a malformed id)", () => {
    expect(permalinkFor("https://x", "a/b c")).toBe("https://x/?run=a%2Fb%20c");
  });

  it("works for a full sha256-hex run id (the real shape)", () => {
    const id = "a".repeat(64);
    expect(permalinkFor("http://localhost:3000", id)).toBe(`http://localhost:3000/?run=${id}`);
  });
});
