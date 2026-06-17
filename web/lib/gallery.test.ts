import { describe, it, expect } from "vitest";
import {
  buildCards,
  groupByShape,
  clipLabel,
  shapeFor,
  displayShape,
  thumbFor,
  hrefFor,
  SHAPE_ORDER,
} from "@/lib/gallery";
import curated from "@/lib/curated.json";
import type { CurationEntry } from "@/lib/types";

// gallery.ts is pure (view-model assembly, grouping, link building — no DOM, no fetch). Mirrors
// program.test.ts. The committed curated.json is the static data source; we assert the helpers
// against it AND against synthetic entries that exercise the defensive branches.

const ENTRIES = curated as CurationEntry[];

describe("buildCards", () => {
  it("builds one view-model per curated entry (defaults to the build-imported curation)", () => {
    const cards = buildCards();
    expect(cards).toHaveLength(ENTRIES.length);
    expect(cards.map((c) => c.queryId)).toEqual(ENTRIES.map((e) => e.id));
    for (const c of cards) {
      expect(c).toHaveProperty("clipLabel");
      expect(c).toHaveProperty("shape");
      expect(c).toHaveProperty("href");
      expect(c).toHaveProperty("thumb");
    }
  });
});

describe("groupByShape", () => {
  it("groups in deterministic SHAPE_ORDER and suppresses empty groups (no hollow Presence)", () => {
    const sections = groupByShape(buildCards());
    // Only shapes that actually exist in the canned set, in SHAPE_ORDER.
    expect(sections.map((s) => s.shape)).toEqual(["count", "first-goal", "scorer-number", "scene"]);
    // no "presence" section even though it is a known eval shape — the canned set has none.
    expect(sections.some((s) => s.shape === "presence")).toBe(false);
    // membership: the two first-goal queries land in the first-goal section.
    const firstGoal = sections.find((s) => s.shape === "first-goal")!;
    expect(firstGoal.cards.map((c) => c.queryId).sort()).toEqual(
      ["bernabeu-23-first-goal", "bernabeu-7-first-goal", "hero-10-first-goal"].sort(),
    );
  });

  it("places unknown shapes last, after the pinned order", () => {
    const cards = buildCards([
      { id: "z", text: "z?", clip: "c", clipLabel: "C", shape: "mystery" },
      { id: "a", text: "a?", clip: "c", clipLabel: "C", shape: "count" },
    ]);
    const sections = groupByShape(cards);
    expect(sections.map((s) => s.shape)).toEqual(["count", "mystery"]);
  });
});

describe("hrefFor / link building", () => {
  it("builds a /?query=<id> link", () => {
    const entry = ENTRIES[0];
    expect(hrefFor(entry)).toBe("/?query=" + encodeURIComponent(entry.id));
    expect(buildCards()[0].href).toBe("/?query=" + encodeURIComponent(ENTRIES[0].id));
  });

  it("URL-encodes an id containing a reserved character", () => {
    const entry: CurationEntry = { id: "a/b?c #d", text: "x", clip: "c", clipLabel: "C", shape: "count" };
    expect(hrefFor(entry)).toBe("/?query=a%2Fb%3Fc%20%23d");
  });
});

describe("shapeFor / displayShape", () => {
  it("returns the lowercase machine shape", () => {
    expect(shapeFor({ id: "x", text: "x", clip: "c", clipLabel: "C", shape: "first-goal" })).toBe("first-goal");
  });

  it("maps machine -> display label", () => {
    expect(displayShape("count")).toBe("Count");
    expect(displayShape("scene")).toBe("Scene");
    expect(displayShape("first-goal")).toBe("First-goal");
    expect(displayShape("scorer-number")).toBe("Scorer-number");
  });

  it("falls back to the raw machine label for an unknown shape", () => {
    expect(displayShape("mystery")).toBe("mystery");
  });
});

describe("clipLabel", () => {
  it("returns the entry's clipLabel", () => {
    expect(clipLabel({ id: "x", text: "x", clip: "single-goal", clipLabel: "Messi vs Mexico — World Cup 2022", shape: "scene" }))
      .toBe("Messi vs Mexico — World Cup 2022");
  });

  it("falls back to the raw clip id when clipLabel is absent (defensive)", () => {
    expect(clipLabel({ id: "x", text: "x", clip: "single-goal", clipLabel: "", shape: "scene" })).toBe("single-goal");
  });
});

describe("thumbFor", () => {
  it("returns a committed still for a pinned single-goal query", () => {
    const t = thumbFor(ENTRIES.find((e) => e.id === "hero-10-first-goal")!);
    expect(t).toEqual({ kind: "img", src: "/frames/goal-4000.jpg" });
    const t2 = thumbFor(ENTRIES.find((e) => e.id === "single-goal-scene")!);
    expect(t2).toEqual({ kind: "img", src: "/frames/scorer-4625.jpg" });
  });

  it("returns a token placeholder for a clip without a committed still", () => {
    const t = thumbFor(ENTRIES.find((e) => e.id === "bernabeu-count-players")!);
    expect(t).toEqual({ kind: "placeholder" });
  });
});

describe("curation consistency / SHAPE_ORDER", () => {
  it("every curated entry's shape is renderable in SHAPE_ORDER (no hollow / unknown shapes)", () => {
    for (const e of ENTRIES) {
      expect(SHAPE_ORDER).toContain(e.shape);
    }
  });
});
