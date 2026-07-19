import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import ClipPlayer from "@/components/ClipPlayer";

// ClipPlayer is the "view the original clip" affordance: a plain <video controls muted playsInline
// preload="metadata"> sourced + postered from lib/frames.ts's manifest. No autoplay (a11y). The
// optional "jump to this step's frame" button seeks currentTime = (focusTs - offset_ms) / 1000 —
// bernabeu's video file starts 7870ms into the source timeline, so a raw source ts would seek to
// the wrong spot without the offset subtraction.

describe("ClipPlayer", () => {
  it("renders nothing for a null/absent clip id (idle, no catalog clip yet)", () => {
    const { container } = render(<ClipPlayer clipId={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing for a clip the manifest doesn't know about", () => {
    const { container } = render(<ClipPlayer clipId="no-such-clip" />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders a video with src/poster/label from the manifest, no autoplay, muted+playsInline+controls", () => {
    render(<ClipPlayer clipId="single-goal" clipLabel="Messi vs Mexico — World Cup 2022" />);
    const video = document.querySelector("video") as HTMLVideoElement;
    expect(video).toBeTruthy();
    expect(video.getAttribute("src")).toBe("/clips/single-goal.mp4");
    expect(video.getAttribute("poster")).toBe("/frames/single-goal/f4625.jpg");
    expect(video).toHaveAttribute("controls");
    // React sets `muted` as a DOM property, not a reflected HTML attribute (documented React
    // quirk for <video>/<audio>) — assert the property, not toHaveAttribute.
    expect(video.muted).toBe(true);
    expect(video).toHaveAttribute("playsinline");
    expect(video).toHaveAttribute("preload", "metadata");
    expect(video).not.toHaveAttribute("autoplay");
    expect(screen.getByText("Messi vs Mexico — World Cup 2022")).toBeInTheDocument();
  });

  it("falls back to the raw clip id as the label when no clipLabel is given", () => {
    render(<ClipPlayer clipId="bernabeu-counter" />);
    expect(screen.getByText("bernabeu-counter")).toBeInTheDocument();
  });

  it("no jump affordance when focusTs is absent/zero", () => {
    render(<ClipPlayer clipId="single-goal" focusTs={0} />);
    expect(screen.queryByRole("button", { name: /jump to this step's frame/i })).toBeNull();
    render(<ClipPlayer clipId="single-goal" />);
    expect(screen.queryByRole("button", { name: /jump to this step's frame/i })).toBeNull();
  });

  it("jump affordance seeks currentTime = (focusTs - offset_ms) / 1000, no throw", () => {
    // bernabeu-counter's video offset_ms is 7870; a step ts of 10000 -> seek to 2.13s into the file.
    render(<ClipPlayer clipId="bernabeu-counter" focusTs={10000} />);
    const btn = screen.getByRole("button", { name: /jump to this step's frame/i });
    const video = document.querySelector("video") as HTMLVideoElement;
    expect(() => btn.click()).not.toThrow();
    expect(video.currentTime).toBeCloseTo((10000 - 7870) / 1000, 5);
  });

  it("jump affordance clamps to 0 when focusTs is before the clip's offset", () => {
    render(<ClipPlayer clipId="bernabeu-counter" focusTs={100} />);
    const btn = screen.getByRole("button", { name: /jump to this step's frame/i });
    const video = document.querySelector("video") as HTMLVideoElement;
    btn.click();
    expect(video.currentTime).toBe(0);
  });
});
