import { describe, it, expect } from "vitest";
import { permalinkUrl } from "@/lib/useRun";

// The copy-link helper is a PURE function (no clipboard, no window side effects) so the URL
// shape can be pinned without a DOM. page.tsx feeds it window.location.origin + meta.run_id.

describe("permalinkUrl", () => {
  it("builds ${origin}/?run=<id> from origin + run_id", () => {
    expect(permalinkUrl("https://demo.example.com", "abc123")).toBe(
      "https://demo.example.com/?run=abc123",
    );
  });

  it("uri-encodes the run id", () => {
    // a real run_id is 64 hex chars (no special chars), but the helper must not break if one
    // slips through — encodeURIComponent keeps the URL well-formed.
    expect(permalinkUrl("http://localhost:3000", "a b/c")).toBe(
      "http://localhost:3000/?run=a%20b%2Fc",
    );
  });

  it("works against a localhost origin (the dev default)", () => {
    const id = "f".repeat(64);
    expect(permalinkUrl("http://localhost:3000", id)).toBe(`http://localhost:3000/?run=${id}`);
  });
});
