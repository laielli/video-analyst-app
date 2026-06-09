import { describe, it, expect } from "vitest";
import { shareLink } from "@/lib/useRun";

// The "Copy run link" affordance copies `${origin}/?run=<id>`. `shareLink` is the pure helper
// behind it (no clipboard, no DOM) so the URL shape is unit-testable in isolation.

describe("shareLink", () => {
  it("builds ${origin}/?run=<id> from origin + run_id", () => {
    expect(shareLink("https://demo.example.com", "abc123")).toBe(
      "https://demo.example.com/?run=abc123",
    );
  });

  it("url-encodes the run id (defense; real ids are hex but the helper must not break)", () => {
    expect(shareLink("https://x.test", "a/b c")).toBe("https://x.test/?run=a%2Fb%20c");
  });

  it("works against a localhost origin", () => {
    expect(shareLink("http://localhost:3000", "deadbeef")).toBe(
      "http://localhost:3000/?run=deadbeef",
    );
  });
});
