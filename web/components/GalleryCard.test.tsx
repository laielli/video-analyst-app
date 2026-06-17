import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import GalleryCard from "@/components/GalleryCard";
import type { GalleryCard as GalleryCardVM } from "@/lib/types";

// GalleryCard is pure presentational. Mirrors StepTracker.test.tsx: render a view-model, assert the
// surfaced content + the anchor + BOTH sides of the thumbnail branch (img vs token placeholder).

const card = (over: Partial<GalleryCardVM> = {}): GalleryCardVM => ({
  queryId: "hero-10-first-goal",
  clipId: "single-goal",
  clipLabel: "Messi vs Mexico — World Cup 2022",
  text: "Does #10 score the first goal?",
  shape: "first-goal",
  href: "/?query=hero-10-first-goal",
  thumb: { kind: "img", src: "/frames/goal-4000.jpg" },
  ...over,
});

describe("GalleryCard", () => {
  it("renders query text, shape tag, and clip label inside a linking anchor", () => {
    render(<GalleryCard card={card()} />);
    expect(screen.getByText("Does #10 score the first goal?")).toBeInTheDocument();
    expect(screen.getByText("First-goal")).toBeInTheDocument();
    expect(screen.getByText("Messi vs Mexico — World Cup 2022")).toBeInTheDocument();
    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("href", "/?query=hero-10-first-goal");
  });

  it("renders the committed <img> for a clip with a still (thumbFor img branch)", () => {
    render(<GalleryCard card={card()} />);
    const img = screen.getByRole("img");
    expect(img).toHaveAttribute("src", "/frames/goal-4000.jpg");
  });

  it("renders a token placeholder (not an <img>) for a clip without a still (thumbFor placeholder branch)", () => {
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
    expect(screen.queryByRole("img")).toBeNull();
    // The placeholder surfaces the clip label + display shape (token-built, no <img>).
    expect(
      screen.getByLabelText("No still for Real Madrid counter — Champions League 2025"),
    ).toBeInTheDocument();
  });
});
