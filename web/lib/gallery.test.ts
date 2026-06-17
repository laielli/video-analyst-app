import { describe, it, expect } from "vitest";
import {
  buildCards,
  groupByShape,
  shapeFor,
  displayShape,
  clipLabel,
  thumbFor,
  SHAPE_ORDER,
} from "@/lib/gallery";
import type { CurationEntry } from "@/lib/types";
import curated from "@/lib/curated.json";

// gallery.ts is pure (it builds view-models + groups from the static curation, no DOM/network).
// Mirrors web/lib/program.test.ts: one case per branch, plus curation-consistency assertions.

const entry = (over: Partial<CurationEntry> = {}): CurationEntry => ({
  id: "x-id",
  text: "Some question?",
  clip: "single-goal",
  clipLabel: "Messi vs Mexico — World Cup 2022",
  shape: "scene",
  ...over,
});

describe("buildCards", () => {
  it("builds a /?query=<id> link, URL-encoded (test #2)", () => {
    const [card] = buildCards([entry({ id: "hero-10-first-goal" })]);
    expect(card.href).toBe("/?query=" + encodeURIComponent("hero-10-first-goal"));
  });

  it("URL-encodes an id containing a reserved char", () => {
    const reserved = "a b/c?d&e";
    const [card] = buildCards([entry({ id: reserved })]);
    expect(card.href).toBe("/?query=" + encodeURIComponent(reserved));
    expect(card.href).toContain("%20"); // space encoded -> proof the encoder ran
  });

  it("carries the view-model fields through (id, clip, label, text, shape)", () => {
    const [card] = buildCards([entry({ id: "q1", clip: "bernabeu-counter", text: "How many?" })]);
    expect(card).toMatchObject({ queryId: "q1", clipId: "bernabeu-counter", text: "How many?" });
  });

  it("defaults to the bundled curation when called with no args (6 cards)", () => {
    expect(buildCards()).toHaveLength(curated.length);
  });
});

describe("groupByShape", () => {
  it("groups in deterministic SHAPE_ORDER and suppresses empty groups (test #1)", () => {
    const sections = groupByShape(buildCards());
    // The canned set has count, first-goal, scorer-number, scene — NO presence section.
    expect(sections.map((s) => s.shape)).toEqual(["count", "first-goal", "scorer-number", "scene"]);
    // No hollow "presence" section even though SHAPE_ORDER is the canonical list.
    expect(sections.some((s) => s.shape === "presence")).toBe(false);
    // first-goal has the 2 bernabeu + 1 hero queries = 3 cards; others have 1 each.
    const firstGoal = sections.find((s) => s.shape === "first-goal")!;
    expect(firstGoal.cards.map((c) => c.queryId).sort()).toEqual(
      ["bernabeu-23-first-goal", "bernabeu-7-first-goal", "hero-10-first-goal"],
    );
  });

  it("sorts unknown shapes last", () => {
    const cards = buildCards([
      entry({ id: "u", shape: "mystery" }),
      entry({ id: "c", shape: "count" }),
    ]);
    const sections = groupByShape(cards);
    expect(sections.map((s) => s.shape)).toEqual(["count", "mystery"]);
  });

  it("pins the canonical SHAPE_ORDER", () => {
    expect(SHAPE_ORDER).toEqual(["count", "first-goal", "scorer-number", "scene"]);
  });
});

describe("shapeFor / displayShape (test #3)", () => {
  it("returns the lowercase machine shape", () => {
    expect(shapeFor(entry({ shape: "first-goal" }))).toBe("first-goal");
  });

  it("maps machine -> display text", () => {
    expect(displayShape("count")).toBe("Count");
    expect(displayShape("scene")).toBe("Scene");
    expect(displayShape("first-goal")).toBe("First-goal");
    expect(displayShape("scorer-number")).toBe("Scorer-number");
  });

  it("falls back to the raw machine label for an unknown shape", () => {
    expect(displayShape("mystery")).toBe("mystery");
  });
});

describe("clipLabel (test #3b)", () => {
  it("returns the entry's clipLabel", () => {
    expect(clipLabel(entry({ clipLabel: "Messi vs Mexico — World Cup 2022" }))).toBe(
      "Messi vs Mexico — World Cup 2022",
    );
  });

  it("falls back to the raw clip id when clipLabel is absent (no crash)", () => {
    expect(clipLabel(entry({ clipLabel: "", clip: "bernabeu-counter" }))).toBe("bernabeu-counter");
  });
});

describe("thumbFor (test #5 — both ternary branches)", () => {
  it("returns a committed still for a pinned single-goal card", () => {
    expect(thumbFor(entry({ id: "hero-10-first-goal" }))).toEqual({
      kind: "img",
      src: "/frames/goal-4000.jpg",
    });
    expect(thumbFor(entry({ id: "single-goal-scene" }))).toEqual({
      kind: "img",
      src: "/frames/scorer-4625.jpg",
    });
  });

  it("falls back to a token placeholder for a clip without a still", () => {
    expect(thumbFor(entry({ id: "bernabeu-count-players" }))).toEqual({ kind: "placeholder" });
  });
});

describe("curation consistency", () => {
  it("every curated entry produces a card whose shape is in SHAPE_ORDER", () => {
    for (const card of buildCards()) {
      expect(SHAPE_ORDER).toContain(card.shape);
    }
  });

  it("groupByShape covers every curated card exactly once", () => {
    const grouped = groupByShape(buildCards()).flatMap((s) => s.cards.map((c) => c.queryId));
    expect(grouped.sort()).toEqual([...curated].map((q) => q.id).sort());
  });
});
