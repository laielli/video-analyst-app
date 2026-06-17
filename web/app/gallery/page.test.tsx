import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import Gallery from "./page";

// The gallery route reads the build-imported static curation directly — NO fetchCatalog, NO
// useEffect/loading state, NO API-down branch (Resolved #4). So unlike page.test.tsx there is no
// catalog stub; instead these tests assert the cards render AND that fetch is never issued.

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Gallery route", () => {
  it("renders a card per curated query (6 cards) from static data", () => {
    render(<Gallery />);
    const links = screen.getAllByRole("link");
    // 6 curated runs + the "Back to analyst" link in the header.
    const cardLinks = links.filter((a) => a.getAttribute("href")?.startsWith("/?query="));
    expect(cardLinks).toHaveLength(6);
  });

  it("renders its full card set offline — fetch is NEVER called", () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    render(<Gallery />);
    // The cards render...
    expect(
      screen.getAllByRole("link").filter((a) => a.getAttribute("href")?.startsWith("/?query=")),
    ).toHaveLength(6);
    // ...without ever touching the network.
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("groups cards under shape sections (deterministic, no hollow presence section)", () => {
    render(<Gallery />);
    // Shape section headings come from the machine->display map.
    expect(screen.getByRole("heading", { name: "Count", level: 2 })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "First-goal", level: 2 })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Scorer-number", level: 2 })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Scene", level: 2 })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Presence", level: 2 })).toBeNull();
  });
});
