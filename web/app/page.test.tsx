import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, waitFor, cleanup, screen } from "@testing-library/react";
import Page from "./page";
import { API_BASE } from "@/lib/useRun";
import { MockEventSource } from "@/vitest.setup";

// Page-load URL-param behavior (Phase 6): a `?run=<id>` permalink auto-replays the stored run,
// `?query=<id>` selects a canned query, and the canned auto-play is GATED so it never clobbers a
// permalink replay. We stub fetchCatalog's network call and drive the on-load effects.

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

const realLocation = window.location;

function stubCatalog() {
  // Only stub fetch — DO NOT vi.unstubAllGlobals() in teardown, or the EventSource global that
  // vitest.setup.ts installs gets nuked and a late passive effect throws "EventSource not defined".
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({ ok: true, json: async () => CATALOG } as Response)),
  );
}

function setSearch(search: string) {
  // jsdom's window.location.search is read-only; replace the whole location with a stub exposing
  // just `.search` and `.origin` (the only fields page.tsx reads).
  Object.defineProperty(window, "location", {
    configurable: true,
    value: { search, origin: "http://localhost:3000" },
  });
}

describe("page load URL params", () => {
  beforeEach(() => {
    stubCatalog();
    setSearch("");
  });
  afterEach(() => {
    cleanup(); // unmount before restoring globals so no late effect constructs an EventSource
    vi.unstubAllGlobals();
    // vi.unstubAllGlobals() also removes the EventSource global vitest.setup.ts installed —
    // reinstall it so other suites (and any straggler effect) still see the mock.
    vi.stubGlobal("EventSource", MockEventSource as unknown as typeof EventSource);
    Object.defineProperty(window, "location", { configurable: true, value: realLocation });
  });

  it("?run=<id> on load replays the stored run (permalink request, not the canned default)", async () => {
    setSearch("?run=abc");
    render(<Page />);
    await waitFor(() => expect(MockEventSource.last).toBeTruthy());
    // the very first (and only) stream is the permalink replay, NOT a canned query.
    expect(MockEventSource.last.url).toBe(`${API_BASE}/api/run?pace_ms=1200&run=abc`);
    // the canned auto-play was gated: exactly one EventSource (no clobbering second run).
    expect(MockEventSource.instances).toHaveLength(1);
  });

  it("?query=<id> on load selects that canned query, not c.queries[0]", async () => {
    setSearch("?query=bernabeu-count-players");
    render(<Page />);
    await waitFor(() => expect(MockEventSource.last).toBeTruthy());
    expect(MockEventSource.last.url).toBe(
      `${API_BASE}/api/run?pace_ms=1200&query=bernabeu-count-players`,
    );
  });

  it("no params -> the first canned query auto-plays", async () => {
    setSearch("");
    render(<Page />);
    await waitFor(() => expect(MockEventSource.last).toBeTruthy());
    expect(MockEventSource.last.url).toBe(
      `${API_BASE}/api/run?pace_ms=1200&query=hero-10-first-goal`,
    );
  });

  it("renders a 'Browse all runs →' link to /gallery in the header (bidirectional nav)", async () => {
    setSearch("");
    render(<Page />);
    const link = await screen.findByRole("link", { name: "Browse all runs →" });
    expect(link).toHaveAttribute("href", "/gallery");
  });
});
