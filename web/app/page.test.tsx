import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, waitFor, act } from "@testing-library/react";
import Page from "@/app/page";
import { API_BASE } from "@/lib/useRun";
import { MockEventSource } from "@/vitest.setup";

// page.tsx reads ?run=<id> (permalink replay) and ?query=<id> (canned selection) off the URL on
// load. These tests stub window.location.search + the catalog fetch and assert which run() the
// page fires (via the EventSource URL the mock records). jsdom has no real network/EventSource.

const CATALOG = {
  clips: [{ id: "single-goal", label: "Messi" }, { id: "bernabeu-counter", label: "Madrid" }],
  queries: [
    { id: "hero-10-first-goal", text: "Does #10 score the first goal?", clip: "single-goal" },
    { id: "bernabeu-count-players", text: "How many players are visible?", clip: "bernabeu-counter" },
  ],
};

function setSearch(search: string) {
  // jsdom: window.location is non-configurable, but `search` can be set via the URL string.
  window.history.replaceState({}, "", `/${search}`);
}

function stubCatalogFetch() {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({ ok: true, json: async () => CATALOG } as Response)),
  );
}

beforeEach(() => {
  // Re-install the EventSource double (vitest.setup.ts stubs it once at module load; our
  // unstubAllGlobals in afterEach would otherwise wipe it for the next test).
  vi.stubGlobal("EventSource", MockEventSource as unknown as typeof EventSource);
  stubCatalogFetch();
});
afterEach(() => {
  vi.unstubAllGlobals();
  setSearch("");
});

describe("Page URL-param load behavior", () => {
  it("?run=<id> on load replays the stored run (permalink), not the default canned query", async () => {
    setSearch("?run=deadbeef");
    await act(async () => {
      render(<Page />);
    });
    await waitFor(() => expect(MockEventSource.last).toBeTruthy());
    // the FIRST EventSource opened is the permalink replay, carrying ?run=.
    expect(MockEventSource.instances[0].url).toBe(`${API_BASE}/api/run?pace_ms=1200&run=deadbeef`);
    // the canned auto-play is gated off, so no &query= run was fired.
    expect(MockEventSource.instances.some((es) => es.url.includes("query=hero-10-first-goal"))).toBe(false);
  });

  it("?query=<id> on load selects that canned query, not c.queries[0]", async () => {
    setSearch("?query=bernabeu-count-players");
    await act(async () => {
      render(<Page />);
    });
    await waitFor(() =>
      expect(MockEventSource.instances.some((es) => es.url.includes("query=bernabeu-count-players"))).toBe(true),
    );
    // it did NOT auto-play the first catalog query.
    expect(MockEventSource.instances.some((es) => es.url.includes("query=hero-10-first-goal"))).toBe(false);
  });

  it("no param on load auto-plays the first canned query (default behavior preserved)", async () => {
    setSearch("");
    await act(async () => {
      render(<Page />);
    });
    await waitFor(() =>
      expect(MockEventSource.instances.some((es) => es.url.includes("query=hero-10-first-goal"))).toBe(true),
    );
  });
});
