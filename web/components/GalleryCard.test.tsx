import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import GalleryCard from "@/components/GalleryCard";
import type { GalleryCard as GalleryCardVM } from "@/lib/types";

// GalleryCard is pure presentational: an <a href> wrapping a thumb (committed still OR token
// placeholder) + query text (Fraunces) + a shape pill + clip label. NO confidence / cached-answer
// preview. Exercises both sides of the thumb ternary (img vs placeholder) for branch coverage.

const card = (partial: Partial<GalleryCardVM>): GalleryCardVM => ({
  queryId: "hero-10-first-goal",
  clipId: "single-goal",
  clipLabel: "Messi vs Mexico — World Cup 2022",
  text: "Does #10 score the first goal?",
  shape: "first-goal",
  href: "/?query=hero-10-first-goal",
  thumb: { kind: "img", src: "/frames/goal-4000.jpg" },
  ...partial,
});

describe("GalleryCard", () => {
  it("renders query text, shape tag, and clip label inside the anchor (Test 4)", () => {
    render(<GalleryCard card={card({})} />);
    expect(screen.getByText("Does #10 score the first goal?")).toBeInTheDocument();
    expect(screen.getByText("First-goal")).toBeInTheDocument();
    expect(screen.getByText("Messi vs Mexico — World Cup 2022")).toBeInTheDocument();
    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("href", "/?query=hero-10-first-goal");
  });

  it("renders a committed <img> for a single-goal card (Test 5 — img branch)", () => {
    render(<GalleryCard card={card({})} />);
    const img = document.querySelector("img") as HTMLImageElement;
    expect(img).toBeTruthy();
    expect(img.getAttribute("src")).toBe("/frames/goal-4000.jpg");
    expect(document.querySelector(".gx-thumb-ph")).toBeNull();
  });

  it("renders the token placeholder (not an <img>) for a clip without a still (Test 5 — placeholder branch)", () => {
    render(
      <GalleryCard
        card={card({
          queryId: "bernabeu-count-players",
          clipId: "bernabeu-counter",
          clipLabel: "Real Madrid counter — Champions League 2025",
          text: "How many players are visible?",
          shape: "count",
          href: "/?query=bernabeu-count-players",
          thumb: { kind: "placeholder" },
        })}
      />,
    );
    expect(document.querySelector("img")).toBeNull();
    const ph = document.querySelector(".gx-thumb-ph") as HTMLElement;
    expect(ph).toBeInTheDocument();
    // placeholder carries the clip label + display shape (token-driven, no still). The clip label
    // appears twice (placeholder + body) and "Count" twice (placeholder + pill) — both are
    // display-mapped from the machine shape.
    expect(screen.getAllByText("Real Madrid counter — Champions League 2025").length).toBeGreaterThanOrEqual(1);
    expect(ph.textContent).toContain("Real Madrid counter — Champions League 2025");
    expect(ph.textContent).toContain("Count");
    expect(screen.getAllByText("Count").length).toBe(2);
  });

  it("uses the view-model href on the anchor (link builder contract)", () => {
    render(<GalleryCard card={card({ href: "/?query=a%20b" })} />);
    expect(screen.getByRole("link")).toHaveAttribute("href", "/?query=a%20b");
  });
});
