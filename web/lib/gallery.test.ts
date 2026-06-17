import { describe, it, expect } from "vitest";
import {
  buildCards,
  groupByShape,
  clipLabel,
  thumbFor,
  shapeFor,
  displayShape,
  SHAPE_ORDER,
} from "@/lib/gallery";
import type { CurationEntry } from "@/lib/types";

// gallery.ts is pure (build-imported curation -> view-models; no DOM, no network). Mirrors
// program.test.ts: one assertion per helper branch, plus curation-consistency checks.

const entry = (over: Partial<CurationEntry> = {}): CurationEntry => ({
  id: "id-x",
  text: "Question?",
  clip: "single-goal",
  clipLabel: "Messi vs Mexico — World Cup 2022",
  shape: "first-goal",
  ...over,
});

describe("buildCards", () => {
  it("builds one view-model per curated entry (defaults to the build-imported curation)", () => {
    // No arg -> reads web/lib/curated.json (the 6 canned queries).
    const cards = buildCards();
    expect(cards).toHaveLength(6);
    for (const c of cards) {
      expect(c.queryId).toBeTruthy();
      expect(c.href).toBe(`/?query=${encodeURIComponent(c.queryId)}`);
    }
  });

  it("carries clip label, shape, and thumbnail onto each card", () => {
    const [card] = buildCards([entry({ id: "hero-10-first-goal" })]);
    expect(card.clipLabel).toBe("Messi vs Mexico — World Cup 2022");
    expect(card.shape).toBe("first-goal");
    expect(card.thumb).toEqual({ kind: "img", src: "/frames/goal-4000.jpg" });
  });

  it("builds a URL-encoded /?query=<id> link (encodes reserved chars)", () => {
    const [card] = buildCards([entry({ id: "a/b c&d" })]);
    expect(card.href).toBe("/?query=" + encodeURIComponent("a/b c&d"));
    expect(card.href).toBe("/?query=a%2Fb%20c%26d");
  });
});

describe("groupByShape", () => {
  it("groups in deterministic SHAPE_ORDER, suppresses empty groups", () => {
    const sections = groupByShape(buildCards());
    // The canned set has count, first-goal, scorer-number, scene — and NO presence section.
    expect(sections.map((s) => s.shape)).toEqual(["count", "first-goal", "scorer-number", "scene"]);
    // No hollow "presence" group even though SHAPE_ORDER does not list it.
    expect(sections.some((s) => s.shape === "presence")).toBe(false);
    // Membership: first-goal holds the 3 first-goal queries (hero + #7 + #23).
    const firstGoal = sections.find((s) => s.shape === "first-goal")!;
    expect(firstGoal.cards.map((c) => c.queryId).sort()).toEqual(
      ["bernabeu-23-first-goal", "bernabeu-7-first-goal", "hero-10-first-goal"],
    );
  });

  it("honors SHAPE_ORDER over scrambled input and sorts unknown shapes last", () => {
    const cards = buildCards([
      entry({ id: "z", shape: "mystery" }),
      entry({ id: "s", shape: "scene" }),
      entry({ id: "c", shape: "count" }),
    ]);
    const sections = groupByShape(cards);
    expect(sections.map((s) => s.shape)).toEqual(["count", "scene", "mystery"]);
  });

  it("SHAPE_ORDER pins the canonical section order", () => {
    expect([...SHAPE_ORDER]).toEqual(["count", "first-goal", "scorer-number", "scene"]);
  });
});

describe("shapeFor / displayShape", () => {
  it("returns the lowercase machine shape", () => {
    expect(shapeFor(entry({ shape: "count" }))).toBe("count");
    expect(shapeFor(entry({ shape: "scene" }))).toBe("scene");
  });

  it("maps machine shape to display text", () => {
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
  it("returns the entry's clip label", () => {
    expect(clipLabel(entry({ clipLabel: "Real Madrid counter — Champions League 2025" })))
      .toBe("Real Madrid counter — Champions League 2025");
  });

  it("falls back to the raw clip id when the label is absent (no crash)", () => {
    expect(clipLabel(entry({ clip: "bernabeu-counter", clipLabel: "" }))).toBe("bernabeu-counter");
  });
});

describe("thumbFor", () => {
  it("returns the committed still for a single-goal card", () => {
    expect(thumbFor(entry({ id: "hero-10-first-goal" }))).toEqual({
      kind: "img",
      src: "/frames/goal-4000.jpg",
    });
    expect(thumbFor(entry({ id: "single-goal-scene" }))).toEqual({
      kind: "img",
      src: "/frames/scorer-4625.jpg",
    });
  });

  it("falls back to a token placeholder for a clip without a committed still", () => {
    expect(thumbFor(entry({ id: "bernabeu-count-players" }))).toEqual({ kind: "placeholder" });
  });
});
