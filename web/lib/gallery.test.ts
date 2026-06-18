import { describe, it, expect } from "vitest";
import {
  buildCards,
  groupByShape,
  clipLabel,
  thumbFor,
  shapeFor,
  hrefFor,
  displayShape,
  SHAPE_ORDER,
} from "@/lib/gallery";
import type { CuratedEntry } from "@/lib/types";

// gallery.ts is pure (it assembles a view-model + groups/links from the build-imported curation,
// no DOM, no fetch). Mirrors program.test.ts: one case per helper / branch.

const SG_FIRST: CuratedEntry = {
  id: "hero-10-first-goal", text: "Does #10 score the first goal?",
  clip: "single-goal", clipLabel: "Messi vs Mexico — World Cup 2022", shape: "first-goal",
};
const BERN_COUNT: CuratedEntry = {
  id: "bernabeu-count-players", text: "How many players are visible?",
  clip: "bernabeu-counter", clipLabel: "Real Madrid counter — Champions League 2025", shape: "count",
};

describe("buildCards", () => {
  it("builds one view-model per curated entry (defaults to the build-imported curation)", () => {
    // No argument -> reads web/lib/curated.json directly (no fetch). The canned set has 6 queries.
    const cards = buildCards();
    expect(cards).toHaveLength(6);
    const ids = cards.map((c) => c.queryId).sort();
    expect(ids).toContain("hero-10-first-goal");
    expect(ids).toContain("bernabeu-scorer-number");
  });

  it("carries text, clipLabel, shape, href and a thumb per card", () => {
    const [card] = buildCards([SG_FIRST]);
    expect(card).toMatchObject({
      queryId: "hero-10-first-goal",
      clipId: "single-goal",
      clipLabel: "Messi vs Mexico — World Cup 2022",
      text: "Does #10 score the first goal?",
      shape: "first-goal",
      href: "/?query=hero-10-first-goal",
    });
    expect(card.thumb).toEqual({ kind: "img", src: "/frames/goal-4000.jpg" });
  });
});

describe("groupByShape — deterministic order, suppresses empty groups", () => {
  it("groups by shape in pinned SHAPE_ORDER and never emits a hollow group", () => {
    const sections = groupByShape(buildCards());
    // The canned set has shapes: count, first-goal, scorer-number, scene (NO presence).
    expect(sections.map((s) => s.shape)).toEqual(["count", "first-goal", "scorer-number", "scene"]);
    // membership: first-goal holds 3 cards (hero #10, bernabeu #7, bernabeu #23).
    const firstGoal = sections.find((s) => s.shape === "first-goal")!;
    expect(firstGoal.cards.map((c) => c.queryId).sort()).toEqual([
      "bernabeu-23-first-goal", "bernabeu-7-first-goal", "hero-10-first-goal",
    ]);
    // no hollow "presence" section (SHAPE_ORDER has no presence; empty groups suppressed anyway).
    expect(sections.some((s) => s.shape === "presence")).toBe(false);
  });

  it("orders known shapes by SHAPE_ORDER and sorts unknown shapes last", () => {
    const cards = buildCards([
      { ...SG_FIRST, id: "x", shape: "mystery" },
      { ...BERN_COUNT },
      { ...SG_FIRST },
    ]);
    const order = groupByShape(cards).map((s) => s.shape);
    // count (rank 0) before first-goal (rank 1) before the unknown "mystery" (last).
    expect(order).toEqual(["count", "first-goal", "mystery"]);
  });

  it("SHAPE_ORDER pins exactly the four shapes that exist in the canned set", () => {
    expect([...SHAPE_ORDER]).toEqual(["count", "first-goal", "scorer-number", "scene"]);
  });
});

describe("hrefFor — URL-encoded /?query= link", () => {
  it("builds /?query=<id> from the curated id", () => {
    expect(hrefFor(SG_FIRST)).toBe("/?query=hero-10-first-goal");
  });

  it("URL-encodes an id containing a reserved char", () => {
    const weird: CuratedEntry = { ...SG_FIRST, id: "a b&c?d" };
    expect(hrefFor(weird)).toBe("/?query=" + encodeURIComponent("a b&c?d"));
    expect(hrefFor(weird)).toBe("/?query=a%20b%26c%3Fd");
  });
});

describe("shapeFor + displayShape", () => {
  it("shapeFor returns the lowercase machine label", () => {
    expect(shapeFor(SG_FIRST)).toBe("first-goal");
    expect(shapeFor(BERN_COUNT)).toBe("count");
  });

  it("displayShape maps machine -> display text (render-time only); unknown passes through", () => {
    expect(displayShape("count")).toBe("Count");
    expect(displayShape("scene")).toBe("Scene");
    expect(displayShape("first-goal")).toBe("First-goal");
    expect(displayShape("scorer-number")).toBe("Scorer-number");
    expect(displayShape("mystery")).toBe("mystery");
  });
});

describe("clipLabel — falls back to raw clip id when missing", () => {
  it("returns the entry's clipLabel when present", () => {
    expect(clipLabel(SG_FIRST)).toBe("Messi vs Mexico — World Cup 2022");
  });

  it("falls back to the raw clip id when clipLabel is absent (no crash)", () => {
    const noLabel = { id: "x", text: "t", clip: "bernabeu-counter", clipLabel: "", shape: "count" } as CuratedEntry;
    expect(clipLabel(noLabel)).toBe("bernabeu-counter");
  });
});

describe("thumbFor — both branches (committed still vs token placeholder)", () => {
  it("returns the committed still for a single-goal query", () => {
    expect(thumbFor(SG_FIRST)).toEqual({ kind: "img", src: "/frames/goal-4000.jpg" });
    expect(thumbFor({ ...SG_FIRST, id: "single-goal-scene" })).toEqual({
      kind: "img", src: "/frames/scorer-4625.jpg",
    });
  });

  it("returns a token placeholder for a bernabeu-counter query (no committed still)", () => {
    expect(thumbFor(BERN_COUNT)).toEqual({ kind: "placeholder" });
  });
});
