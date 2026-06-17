import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import Gallery from "./page";

// The gallery reads the build-imported static curation (web/lib/curated.json) — NO fetchCatalog,
// NO network. Mirrors page.test.tsx's render idiom, but ASSERTS the gallery does NOT fetch
// (Resolved #4): a credential-less / API-down front door must still render the full card set.

afterEach(() => cleanup());

describe("Gallery route", () => {
  it("renders a card per curated query from static data (6 cards)", () => {
    render(<Gallery />);
    const links = screen.getAllByRole("link");
    // 6 cards + the "Open the analyst" back link in the header.
    const cardLinks = links.filter((a) => (a.getAttribute("href") ?? "").startsWith("/?query="));
    expect(cardLinks).toHaveLength(6);
    // every card links into the analyst's canned load path.
    for (const a of cardLinks) {
      expect(a.getAttribute("href")).toMatch(/^\/\?query=/);
    }
  });

  it("renders the full card set offline — fetch is NEVER called", () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    try {
      render(<Gallery />);
      const cardLinks = screen
        .getAllByRole("link")
        .filter((a) => (a.getAttribute("href") ?? "").startsWith("/?query="));
      expect(cardLinks).toHaveLength(6);
      // the front door has ZERO runtime API dependency.
      expect(fetchSpy).not.toHaveBeenCalled();
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("groups cards under shape sections in deterministic order with no hollow group", () => {
    render(<Gallery />);
    const headings = screen.getAllByRole("heading", { level: 2 }).map((h) => h.textContent);
    expect(headings).toEqual(["Count", "First-goal", "Scorer-number", "Scene"]);
  });

  it("links bidirectionally back to the analyst", () => {
    render(<Gallery />);
    const back = screen.getByRole("link", { name: /Open the analyst/i });
    expect(back).toHaveAttribute("href", "/");
  });
});
