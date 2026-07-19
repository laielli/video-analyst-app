import { describe, it, expect } from "vitest";
import { frameSrc, posterSrc, clipFrames, nearestTs } from "@/lib/frames";

// frames.ts is pure (build-imported frames-manifest.json, no DOM/fetch). frameSrc is clip-scoped
// (the root-cause fix: a step's frame can only ever come from ITS clip) and snaps to the nearest
// sampled ts, ties resolving to the earlier frame.

describe("nearestTs", () => {
  it("snaps to the exact ts when present", () => {
    expect(nearestTs([100, 200, 300], 200)).toBe(200);
  });

  it("snaps to the closer neighbor", () => {
    expect(nearestTs([100, 200, 300], 240)).toBe(200);
    expect(nearestTs([100, 200, 300], 260)).toBe(300);
  });

  it("ties resolve to the earlier (smaller) ts", () => {
    expect(nearestTs([100, 200], 150)).toBe(100);
  });

  it("empty list -> null", () => {
    expect(nearestTs([], 100)).toBeNull();
  });
});

describe("frameSrc", () => {
  it("null/absent ts -> null (no meaningful frame)", () => {
    expect(frameSrc("single-goal", 0)).toBeNull();
    expect(frameSrc("single-goal", undefined)).toBeNull();
    expect(frameSrc("single-goal", null)).toBeNull();
  });

  it("null/absent clip -> null", () => {
    expect(frameSrc(null, 4000)).toBeNull();
    expect(frameSrc(undefined, 4000)).toBeNull();
  });

  it("unknown clip -> null (never falls back to another clip's stills)", () => {
    expect(frameSrc("no-such-clip", 4000)).toBeNull();
  });

  it("single-goal: an exact sampled ts resolves to its still", () => {
    expect(frameSrc("single-goal", 4000)).toBe("/frames/single-goal/f4000.jpg");
    expect(frameSrc("single-goal", 4625)).toBe("/frames/single-goal/f4625.jpg");
  });

  it("single-goal: an in-between ts snaps to the nearest sampled still", () => {
    // 4050 is 50ms from 4000 and 75ms from 4125 -> nearest is 4000.
    expect(frameSrc("single-goal", 4050)).toBe("/frames/single-goal/f4000.jpg");
  });

  it("single-goal: a ts past either edge clamps to the nearest edge still", () => {
    expect(frameSrc("single-goal", 0 + 1)).toBe("/frames/single-goal/f3500.jpg"); // 1ms, still truthy
    expect(frameSrc("single-goal", 999999)).toBe("/frames/single-goal/f5000.jpg");
  });

  it("bernabeu-counter: resolves within its own (later, source-timeline) ts range", () => {
    // bernabeu's stills start at 9000 — far outside single-goal's 3500-5000 range. A bernabeu step
    // ts must resolve against bernabeu's OWN stills, never the hero's.
    expect(frameSrc("bernabeu-counter", 10000)).toBe("/frames/bernabeu-counter/f10000.jpg");
    expect(frameSrc("bernabeu-counter", 14375)).toBe("/frames/bernabeu-counter/f14375.jpg");
  });

  it("bernabeu-counter: an in-between ts snaps to the nearest 125ms-sampled still", () => {
    expect(frameSrc("bernabeu-counter", 14400)).toBe("/frames/bernabeu-counter/f14375.jpg");
  });
});

describe("clipFrames", () => {
  it("returns the manifest entry for a known clip", () => {
    const entry = clipFrames("bernabeu-counter");
    expect(entry).not.toBeNull();
    expect(entry?.poster_ts).toBe(10000);
    expect(entry?.video).toEqual({ src: "/clips/bernabeu-counter.mp4", offset_ms: 7870, duration_ms: 6950 });
  });

  it("returns null for an unknown clip", () => {
    expect(clipFrames("no-such-clip")).toBeNull();
  });
});

describe("posterSrc", () => {
  it("builds the poster still URL from poster_ts", () => {
    expect(posterSrc("single-goal")).toBe("/frames/single-goal/f4625.jpg");
    expect(posterSrc("bernabeu-counter")).toBe("/frames/bernabeu-counter/f10000.jpg");
  });

  it("unknown clip -> null", () => {
    expect(posterSrc("no-such-clip")).toBeNull();
  });
});
