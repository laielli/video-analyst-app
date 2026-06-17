import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import Gallery from "./page";
import curated from "@/lib/curated.json";

// The gallery route reads the build-imported static curation (Resolved #4) — NO fetchCatalog, NO
// useEffect/loading state. So unlike page.test.tsx, the route test asserts render WITHOUT a network
// call. Mirrors web/app/page.test.tsx render idiom, inverted (asserts fetch is NEVER called).

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Gallery route", () => {
  it("renders one card link per curated query, from static data (test #6)", () => {
    render(<Gallery />);
    const links = screen.getAllByRole("link");
    // One link per curated entry PLUS the header "← Open the analyst" back-link.
    const cardLinks = links.filter((a) => a.getAttribute("href")?.startsWith("/?query="));
    expect(cardLinks).toHaveLength(curated.length);
    // Every curated id is reachable from a card link.
    const hrefs = cardLinks.map((a) => a.getAttribute("href"));
    for (const q of curated) {
      expect(hrefs).toContain("/?query=" + encodeURIComponent(q.id));
    }
  });

  it("renders its full card set offline — fetch is NEVER issued (test #7)", () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    render(<Gallery />);
    // Cards rendered...
    const cardLinks = screen
      .getAllByRole("link")
      .filter((a) => a.getAttribute("href")?.startsWith("/?query="));
    expect(cardLinks).toHaveLength(curated.length);
    // ...and NO network call was ever made (zero runtime API dependency).
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("groups cards under shape section headings (no empty presence section)", () => {
    render(<Gallery />);
    // The four shapes present in the canned set render as section headings.
    expect(screen.getByRole("heading", { name: "Count", level: 2 })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "First-goal", level: 2 })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Scorer-number", level: 2 })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Scene", level: 2 })).toBeInTheDocument();
    // No hollow presence section.
    expect(screen.queryByRole("heading", { name: "Presence", level: 2 })).toBeNull();
  });
});
