import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, waitFor, cleanup } from "@testing-library/react";
import { MockEventSource } from "@/vitest.setup";
import { API_BASE } from "@/lib/useRun";

// Page-level URL-param parsing: on load, ?run=<id> replays the stored run (a permalink request),
// ?query=<id> selects that canned catalog query, and a bare load auto-plays the first canned query.
// We mock fetchCatalog (jsdom has no network) and set window.location.search per test, then assert
// on the EventSource URL the page opened. We do NOT vi.unstubAllGlobals() — that would also remove
// vitest.setup.ts's EventSource stub, which the page constructs.

const CATALOG = {
  clips: [{ id: "single-goal", label: "Messi" }],
  queries: [
    { id: "hero-10-first-goal", text: "Does #10 score the first goal?", clip: "single-goal" },
    { id: "bernabeu-count-players", text: "How many players are visible?", clip: "bernabeu-counter" },
  ],
};

const origFetch = globalThis.fetch;

function setSearch(search: string) {
  // Override only window.location.search (leave the rest of Location intact).
  Object.defineProperty(window, "location", {
    configurable: true,
    value: Object.assign(Object.create(Object.getPrototypeOf(window.location)), window.location, {
      search,
      origin: "http://localhost:3000",
    }),
  });
}

async function renderPage() {
  const { default: Page } = await import("@/app/page");
  return render(<Page />);
}

describe("Page URL-param auto-play", () => {
  beforeEach(() => {
    vi.resetModules();
    globalThis.fetch = vi.fn(async () => ({ ok: true, json: async () => CATALOG } as Response)) as typeof fetch;
  });

  afterEach(() => {
    cleanup(); // unmount so trailing effects don't construct EventSources after teardown
    globalThis.fetch = origFetch;
    setSearch("");
    MockEventSource.reset();
  });

  it("?run=<id> on load replays the stored run (permalink request, not the canned default)", async () => {
    setSearch("?run=abcdef");
    await renderPage();
    await waitFor(() => expect(MockEventSource.instances.length).toBeGreaterThan(0));
    // the FIRST stream opened is the permalink replay.
    expect(MockEventSource.instances[0].url).toBe(`${API_BASE}/api/run?pace_ms=1200&run=abcdef`);
    // and the canned auto-play is gated off: no query= stream is opened.
    expect(MockEventSource.instances.every((s) => !s.url.includes("query="))).toBe(true);
  });

  it("?query=<id> on load selects that canned query (not c.queries[0])", async () => {
    setSearch("?query=bernabeu-count-players");
    await renderPage();
    await waitFor(() =>
      expect(MockEventSource.instances.some((s) => s.url.includes("query="))).toBe(true),
    );
    const cannedStream = MockEventSource.instances.find((s) => s.url.includes("query="))!;
    expect(cannedStream.url).toContain("query=bernabeu-count-players");
    expect(cannedStream.url).not.toContain("hero-10-first-goal");
  });

  it("bare load auto-plays the first canned query", async () => {
    setSearch("");
    await renderPage();
    await waitFor(() =>
      expect(MockEventSource.instances.some((s) => s.url.includes("query="))).toBe(true),
    );
    const cannedStream = MockEventSource.instances.find((s) => s.url.includes("query="))!;
    expect(cannedStream.url).toContain("query=hero-10-first-goal");
  });
});
