import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import GalleryCard from "@/components/GalleryCard";
import type { GalleryCardVM } from "@/lib/types";

// GalleryCard renders an anchor (real navigation — survives static export) wrapping the thumbnail
// + query text + a shape tag + clip label. Pure presentational. We exercise BOTH sides of
// thumbFor's outcome via the view-model's thumb (img vs placeholder).

const imgCard: GalleryCardVM = {
  queryId: "hero-10-first-goal",
  clipId: "single-goal",
  clipLabel: "Messi vs Mexico — World Cup 2022",
  text: "Does #10 score the first goal?",
  shape: "first-goal",
  href: "/?query=hero-10-first-goal",
  thumb: { kind: "img", src: "/frames/goal-4000.jpg" },
};

const placeholderCard: GalleryCardVM = {
  queryId: "bernabeu-count-players",
  clipId: "bernabeu-counter",
  clipLabel: "Real Madrid counter — Champions League 2025",
  text: "How many players are visible?",
  shape: "count",
  href: "/?query=bernabeu-count-players",
  thumb: { kind: "placeholder" },
};

describe("GalleryCard", () => {
  it("renders query text + shape tag + clip label inside an anchor whose href matches the VM", () => {
    render(<GalleryCard card={imgCard} />);
    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("href", imgCard.href);
    expect(link).toHaveTextContent("Does #10 score the first goal?");
    // machine shape "first-goal" renders as the display label "First-goal".
    expect(link).toHaveTextContent("First-goal");
    expect(link).toHaveTextContent("Messi vs Mexico — World Cup 2022");
  });

  it("renders the committed <img> for a single-goal card (one side of the thumb branch)", () => {
    const { container } = render(<GalleryCard card={imgCard} />);
    // alt="" makes the still presentational (no img role), so query the element directly.
    const img = container.querySelector("img.gx-thumb-img");
    expect(img).not.toBeNull();
    expect(img).toHaveAttribute("src", "/frames/goal-4000.jpg");
    expect(container.querySelector(".gx-thumb-placeholder")).toBeNull();
  });

  it("renders a token placeholder (not an <img>) for a clip without a still (other branch)", () => {
    const { container } = render(<GalleryCard card={placeholderCard} />);
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector(".gx-thumb-placeholder")).not.toBeNull();
    // the placeholder surfaces the clip label + display shape.
    const link = screen.getByRole("link");
    expect(link).toHaveTextContent("Real Madrid counter — Champions League 2025");
    expect(link).toHaveTextContent("Count");
  });
});
