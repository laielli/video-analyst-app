import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, waitFor } from "@testing-library/react";
import { API_BASE } from "@/lib/useRun";
import { MockEventSource } from "@/vitest.setup";

// Page-load behavior for the shareable-permalink story. The page reads ?run / ?query off the URL
// on mount: ?run replays the stored run (NOT the canned auto-play), ?query selects that canned
// query, and neither falls back to c.queries[0]. We mock fetchCatalog (jsdom has no network) and
// stub window.location.search, then inspect which EventSource URL the page opened.

const catalog = {
  clips: [{ id: "single-goal", label: "Messi" }],
  queries: [
    { id: "hero-10-first-goal", text: "Does #10 score the first goal?", clip: "single-goal" },
    { id: "bernabeu-count-players", text: "How many players are visible?", clip: "bernabeu-counter" },
  ],
};

// Mock the catalog fetch so the auto-play effect has queries; useRun stays REAL (so the
// MockEventSource records the URL the page opened).
vi.mock("@/lib/useRun", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/useRun")>();
  return { ...actual, fetchCatalog: vi.fn(async () => catalog) };
});

import Page from "@/app/page";

function setSearch(search: string) {
  Object.defineProperty(window, "location", {
    value: { ...window.location, search, origin: "https://demo.example" },
    writable: true,
    configurable: true,
  });
}

const lastUrl = () => MockEventSource.last?.url;

beforeEach(() => {
  MockEventSource.reset();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("Page URL-param load", () => {
  it("?run=<id> on load replays the stored run (permalink), not the default canned query", async () => {
    setSearch("?run=abc123");
    render(<Page />);
    await waitFor(() => expect(MockEventSource.last).toBeTruthy());
    expect(lastUrl()).toContain("run=abc123");
    expect(lastUrl()).not.toContain("query=");
  });

  it("?query=<id> on load selects that canned query, not c.queries[0]", async () => {
    setSearch("?query=bernabeu-count-players");
    render(<Page />);
    await waitFor(() => expect(MockEventSource.last).toBeTruthy());
    expect(lastUrl()).toContain("query=bernabeu-count-players");
  });

  it("no params -> the first canned query auto-plays (default behavior preserved)", async () => {
    setSearch("");
    render(<Page />);
    await waitFor(() => expect(MockEventSource.last).toBeTruthy());
    expect(lastUrl()).toContain("query=hero-10-first-goal");
  });

  it("an unknown ?query=<id> falls back to the first canned query", async () => {
    setSearch("?query=does-not-exist");
    render(<Page />);
    await waitFor(() => expect(MockEventSource.last).toBeTruthy());
    expect(lastUrl()).toContain("query=hero-10-first-goal");
  });
});
