import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import Gallery from "./page";
import curated from "@/lib/curated.json";

// The gallery's data source is the build-imported static curation (Resolved #4) — so unlike
// page.test.tsx there is NO fetchCatalog stub: the route test asserts render WITHOUT a network
// call. Mirrors page.test.tsx's render posture, inverted (assert fetch was NOT called).

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Gallery route", () => {
  it("renders a card per curated query, from static data, with NO network call (Test 6)", () => {
    render(<Gallery />);
    // one link per curated entry (6 cards).
    const links = screen.getAllByRole("link");
    // the header carries one extra "Open the analyst" nav link, so filter to the ?query= cards.
    const cardLinks = links.filter((a) => a.getAttribute("href")?.startsWith("/?query="));
    expect(cardLinks).toHaveLength(curated.length);
    expect(cardLinks).toHaveLength(6);
  });

  it("renders the full card set offline — fetch is NEVER issued (Test 7)", () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    render(<Gallery />);
    // cards render...
    expect(screen.getByText("Does #10 score the first goal?")).toBeInTheDocument();
    expect(screen.getByText("How many players are visible?")).toBeInTheDocument();
    // ...and NO network call was made (the curation is build-imported).
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("groups cards into shape sections in deterministic order", () => {
    render(<Gallery />);
    const headings = screen.getAllByRole("heading", { level: 2 }).map((h) => h.textContent);
    expect(headings).toEqual(["Count", "First-goal", "Scorer-number", "Scene"]);
  });
});
