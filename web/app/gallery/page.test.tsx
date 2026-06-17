import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import Gallery from "./page";
import curated from "@/lib/curated.json";

// The gallery route reads the build-imported static curation — NO fetchCatalog, NO useEffect,
// NO loading/API-down branch (Resolved #4). So unlike page.test.tsx there is NO fetch stub: we
// render <Gallery /> directly and assert it renders a card per curated query AND never issues a
// network call (the offline-front-door guarantee).

describe("Gallery route", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("renders one link (card) per curated query from static data — no network call", () => {
    render(<Gallery />);
    const links = screen.getAllByRole("link");
    // every curated query has a card linking into /?query=<id>...
    for (const entry of curated) {
      expect(screen.getByRole("link", { name: `Open analyst: ${entry.text}` }))
        .toHaveAttribute("href", `/?query=${encodeURIComponent(entry.id)}`);
    }
    // ...plus the "Back to analyst" nav link in the header. So at least curated.length cards.
    const cardLinks = links.filter((l) => l.getAttribute("href")?.startsWith("/?query="));
    expect(cardLinks).toHaveLength(curated.length);
  });

  it("renders its full card set offline — fetch is NEVER issued", () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    render(<Gallery />);
    // cards rendered...
    const cardLinks = screen.getAllByRole("link").filter((l) => l.getAttribute("href")?.startsWith("/?query="));
    expect(cardLinks).toHaveLength(curated.length);
    // ...and no network call was made.
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});
