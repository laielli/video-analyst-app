import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import GalleryCard from "@/components/GalleryCard";
import type { GalleryCardVM } from "@/lib/types";

// GalleryCard is pure presentational: a real <a> into the analyst wrapping a thumbnail + query
// text + a shape tag + clip label. Mirrors web/components/StepTracker.test.tsx render idiom.

function vm(over: Partial<GalleryCardVM> = {}): GalleryCardVM {
  return {
    queryId: "hero-10-first-goal",
    clipId: "single-goal",
    clipLabel: "Messi vs Mexico — World Cup 2022",
    text: "Does #10 score the first goal?",
    shape: "first-goal",
    href: "/?query=hero-10-first-goal",
    thumb: { kind: "img", src: "/frames/goal-4000.jpg" },
    ...over,
  };
}

describe("GalleryCard (test #4)", () => {
  it("renders query text, shape tag, and clip label inside an anchor whose href matches the VM", () => {
    render(<GalleryCard card={vm()} />);
    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("href", "/?query=hero-10-first-goal");
    expect(link).toHaveTextContent("Does #10 score the first goal?");
    expect(link).toHaveTextContent("First-goal"); // machine -> display mapped shape tag
    expect(link).toHaveTextContent("Messi vs Mexico — World Cup 2022");
  });

  it("renders a committed <img> for a single-goal card (thumbFor img branch)", () => {
    render(<GalleryCard card={vm()} />);
    const img = screen.getByRole("img");
    expect(img).toHaveAttribute("src", "/frames/goal-4000.jpg");
  });
});

describe("GalleryCard thumbnail placeholder (test #5 — placeholder branch)", () => {
  it("renders the token placeholder (not an <img>) for a clip without a still", () => {
    render(
      <GalleryCard
        card={vm({
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
    expect(screen.queryByRole("img")).toBeNull();
    expect(screen.getByLabelText("No still for this clip")).toBeInTheDocument();
    // The placeholder carries the clip label + display shape (token-built, no <img>).
    expect(screen.getByLabelText("No still for this clip")).toHaveTextContent("Count");
  });
});
