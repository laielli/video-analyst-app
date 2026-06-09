import { describe, it, expect } from "vitest";
import { buildRunLink, parseLoadParams } from "@/lib/permalink";

// Pure helpers behind the shareable-permalink affordance. No clipboard, no window — these are the
// load-bearing string/URL functions that page.tsx wires to the real navigator.clipboard +
// window.location.

describe("buildRunLink", () => {
  it("copy-link is built from origin + run_id", () => {
    expect(buildRunLink("https://demo.example", "abc123")).toBe(
      "https://demo.example/?run=abc123",
    );
  });

  it("encodes the run id", () => {
    expect(buildRunLink("https://x", "a/b c")).toBe("https://x/?run=a%2Fb%20c");
  });
});

describe("parseLoadParams", () => {
  it("extracts ?run=<id>", () => {
    expect(parseLoadParams("?run=deadbeef")).toEqual({ run: "deadbeef", query: null });
  });

  it("extracts ?query=<id>", () => {
    expect(parseLoadParams("?query=hero-10-first-goal")).toEqual({
      run: null,
      query: "hero-10-first-goal",
    });
  });

  it("returns nulls for an empty search string", () => {
    expect(parseLoadParams("")).toEqual({ run: null, query: null });
  });

  it("surfaces both when present (page.tsx lets run win)", () => {
    expect(parseLoadParams("?run=abc&query=hero")).toEqual({ run: "abc", query: "hero" });
  });
});
