import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, waitFor, cleanup } from "@testing-library/react";
import Page from "@/app/page";
import { API_BASE } from "@/lib/useRun";
import { MockEventSource } from "@/vitest.setup";

// Page-level URL-param load behavior: a shared ?run=<id> link replays the stored run; ?query=<id>
// selects that canned catalog query; a bare load auto-plays the first canned query. We mock the
// catalog fetch (jsdom has no network) and drive the page through the global MockEventSource.
//
// NOTE: the URL search is set via history.replaceState (jsdom supports it) so we never redefine
// window.location wholesale — and we restore `fetch` directly rather than vi.unstubAllGlobals(),
// which would also tear down the EventSource stub installed once in vitest.setup.ts and break any
// effects that fire after the test resolves.

const CATALOG = {
  clips: [
    { id: "single-goal", label: "Messi vs Mexico" },
    { id: "bernabeu-counter", label: "Real Madrid counter" },
  ],
  queries: [
    { id: "hero-10-first-goal", text: "Does #10 score the first goal?", clip: "single-goal" },
    { id: "bernabeu-count-players", text: "How many players are visible?", clip: "bernabeu-counter" },
  ],
};

const realFetch = globalThis.fetch;

beforeEach(() => {
  globalThis.fetch = vi.fn(async () => ({ ok: true, json: async () => CATALOG }) as Response);
});

afterEach(() => {
  cleanup();
  globalThis.fetch = realFetch;
  window.history.replaceState({}, "", "/");
});

describe("Page URL-param load", () => {
  it("?run=<id> on load replays the stored run (permalink request, not the canned query)", async () => {
    const runId = "a".repeat(64);
    window.history.replaceState({}, "", `/?run=${runId}`);
    render(<Page />);
    await waitFor(() => expect(MockEventSource.instances.length).toBeGreaterThan(0));
    // the permalink replay is opened (independent of the catalog) and no canned auto-play fires.
    expect(MockEventSource.instances[0].url).toBe(`${API_BASE}/api/run?pace_ms=1200&run=${runId}`);
    expect(MockEventSource.instances.every((s) => !s.url.includes("&query="))).toBe(true);
  });

  it("?query=<id> on load selects that canned query (not c.queries[0])", async () => {
    window.history.replaceState({}, "", "/?query=bernabeu-count-players");
    render(<Page />);
    await waitFor(() =>
      expect(MockEventSource.instances.some((s) => s.url.includes("query=bernabeu-count-players"))).toBe(true),
    );
    expect(MockEventSource.last.url).toBe(
      `${API_BASE}/api/run?pace_ms=1200&query=bernabeu-count-players`,
    );
  });

  it("a bare load auto-plays the first canned query", async () => {
    window.history.replaceState({}, "", "/");
    render(<Page />);
    await waitFor(() =>
      expect(MockEventSource.instances.some((s) => s.url.includes("query=hero-10-first-goal"))).toBe(true),
    );
    expect(MockEventSource.last.url).toBe(
      `${API_BASE}/api/run?pace_ms=1200&query=hero-10-first-goal`,
    );
  });
});
