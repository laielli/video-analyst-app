import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import GalleryCard from "@/components/GalleryCard";
import type { GalleryCardVM } from "@/lib/types";

// GalleryCard is pure presentational. It renders query text (Fraunces), a shape tag, and the clip
// label inside an anchor to /?query=<id>. Thumbnail exercises BOTH branches of thumbFor: a
// committed <img> for single-goal, a token placeholder (NOT an <img>) for bernabeu-counter.

function card(partial: Partial<GalleryCardVM> = {}): GalleryCardVM {
  return {
    queryId: "hero-10-first-goal",
    clipId: "single-goal",
    clipLabel: "Messi vs Mexico — World Cup 2022",
    text: "Does #10 score the goal?",
    shape: "first-goal",
    href: "/?query=hero-10-first-goal",
    thumb: { kind: "img", src: "/frames/goal-4000.jpg" },
    ...partial,
  };
}

describe("GalleryCard", () => {
  it("renders the query text, the shape tag, and the clip label inside an anchor", () => {
    render(<GalleryCard card={card()} />);
    expect(screen.getByText("Does #10 score the goal?")).toBeInTheDocument();
    // machine shape -> display text
    expect(screen.getByText("Goal")).toBeInTheDocument();
    expect(screen.getByText("Messi vs Mexico — World Cup 2022")).toBeInTheDocument();
    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("href", "/?query=hero-10-first-goal");
  });

  it("renders a committed <img> for a clip that has a still (img branch)", () => {
    const { container } = render(<GalleryCard card={card()} />);
    const img = container.querySelector("img.gx-thumb-img");
    expect(img).not.toBeNull();
    expect(img).toHaveAttribute("src", "/frames/goal-4000.jpg");
    expect(container.querySelector(".gx-thumb-ph")).toBeNull();
  });

  it("renders a token placeholder (not an <img>) for a clip with no still (placeholder branch)", () => {
    const { container } = render(
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
    expect(container.querySelector(".gx-thumb-ph")).not.toBeNull();
    expect(container.querySelector("img")).toBeNull();
    // the placeholder surfaces the clip label + shape (token-driven, no still)
    expect(screen.getAllByText("Real Madrid counter — Champions League 2025").length).toBeGreaterThan(0);
    // "Count" appears in both the placeholder and the shape tag.
    expect(screen.getAllByText("Count").length).toBeGreaterThanOrEqual(1);
  });
});
