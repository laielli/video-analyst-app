import { describe, it, expect } from "vitest";
import {
  buildCards,
  groupByShape,
  clipLabel,
  thumbFor,
  shapeFor,
  displayShape,
  linkFor,
  SHAPE_ORDER,
} from "@/lib/gallery";
import type { CurationEntry } from "@/lib/types";

// gallery.ts is pure (it builds card view-models + groups them from the build-imported static
// curation — no DOM, no fetch). Mirrors program.test.ts: one assertion per helper / branch.

const entry = (partial: Partial<CurationEntry>): CurationEntry => ({
  id: "x",
  text: "?",
  clip: "single-goal",
  clipLabel: "Messi vs Mexico — World Cup 2022",
  shape: "first-goal",
  ...partial,
});

describe("groupByShape", () => {
  it("groups in deterministic SHAPE_ORDER, suppressing empty groups (Test 1)", () => {
    // shapes present in the canned set: count, first-goal, scorer-number, scene (NO presence).
    const cards = buildCards();
    const sections = groupByShape(cards);
    // section ORDER follows SHAPE_ORDER over the shapes that exist; no hollow "presence" section.
    expect(sections.map((s) => s.shape)).toEqual(["count", "first-goal", "scorer-number", "scene"]);
    // membership: first-goal carries the 3 first-goal queries.
    const fg = sections.find((s) => s.shape === "first-goal")!;
    expect(fg.cards.map((c) => c.queryId).sort()).toEqual(
      ["bernabeu-23-first-goal", "bernabeu-7-first-goal", "hero-10-first-goal"],
    );
    expect(sections.find((s) => s.shape === "count")!.cards).toHaveLength(1);
  });

  it("sorts unknown shapes last (stable)", () => {
    const cards = [
      { queryId: "u", clipId: "c", clipLabel: "C", text: "?", shape: "mystery", href: "/", thumb: { kind: "placeholder" } as const },
      { queryId: "s", clipId: "c", clipLabel: "C", text: "?", shape: "scene", href: "/", thumb: { kind: "placeholder" } as const },
      { queryId: "n", clipId: "c", clipLabel: "C", text: "?", shape: "count", href: "/", thumb: { kind: "placeholder" } as const },
    ];
    expect(groupByShape(cards).map((s) => s.shape)).toEqual(["count", "scene", "mystery"]);
  });

  it("emits NO presence section (SHAPE_ORDER omits a shape the canned set lacks)", () => {
    expect(SHAPE_ORDER).not.toContain("presence");
    expect(groupByShape(buildCards()).map((s) => s.shape)).not.toContain("presence");
  });
});

describe("linkFor / buildCards href", () => {
  it("builds a /?query=<id> link, URL-encoded (Test 2)", () => {
    const q = entry({ id: "hero-10-first-goal" });
    expect(linkFor(q)).toBe("/?query=" + encodeURIComponent("hero-10-first-goal"));
  });

  it("URL-encodes an id containing a reserved char (Test 2)", () => {
    const q = entry({ id: "a b&c=d" });
    expect(linkFor(q)).toBe("/?query=" + encodeURIComponent("a b&c=d"));
    expect(linkFor(q)).toContain("a%20b%26c%3Dd");
  });

  it("buildCards wires href off linkFor for every entry", () => {
    for (const c of buildCards()) {
      expect(c.href).toBe("/?query=" + encodeURIComponent(c.queryId));
    }
  });
});

describe("shapeFor / displayShape", () => {
  it("returns the lowercase machine shape and maps to display text (Test 3)", () => {
    expect(shapeFor(entry({ shape: "count" }))).toBe("count");
    expect(displayShape("count")).toBe("Count");
    expect(displayShape("scene")).toBe("Scene");
    expect(displayShape("first-goal")).toBe("First-goal");
    expect(displayShape("scorer-number")).toBe("Scorer-number");
  });

  it("displayShape falls back to the raw label for an unmapped shape", () => {
    expect(displayShape("mystery")).toBe("mystery");
  });
});

describe("clipLabel", () => {
  it("returns the entry's clipLabel (Test 3b)", () => {
    expect(clipLabel(entry({ clipLabel: "Real Madrid counter — Champions League 2025" }))).toBe(
      "Real Madrid counter — Champions League 2025",
    );
  });

  it("falls back to the raw clip id when clipLabel is absent (Test 3b)", () => {
    expect(clipLabel(entry({ clipLabel: undefined, clip: "bernabeu-counter" }))).toBe("bernabeu-counter");
  });
});

describe("thumbFor", () => {
  it("returns the committed still for a pinned single-goal query (Test 5 — img branch)", () => {
    expect(thumbFor(entry({ id: "hero-10-first-goal" }))).toEqual({ kind: "img", src: "/frames/goal-4000.jpg" });
    expect(thumbFor(entry({ id: "single-goal-scene" }))).toEqual({ kind: "img", src: "/frames/scorer-4625.jpg" });
  });

  it("returns a placeholder for a bernabeu-counter query (Test 5 — placeholder branch)", () => {
    expect(thumbFor(entry({ id: "bernabeu-count-players", clip: "bernabeu-counter" }))).toEqual({ kind: "placeholder" });
  });
});

describe("buildCards", () => {
  it("builds one fully-resolved card per curated entry", () => {
    const cards = buildCards();
    expect(cards).toHaveLength(6);
    const hero = cards.find((c) => c.queryId === "hero-10-first-goal")!;
    expect(hero).toMatchObject({
      clipId: "single-goal",
      clipLabel: "Messi vs Mexico — World Cup 2022",
      shape: "first-goal",
      thumb: { kind: "img", src: "/frames/goal-4000.jpg" },
    });
  });

  it("accepts an injected curation (default-param branch)", () => {
    const cards = buildCards([entry({ id: "only", shape: "count" })]);
    expect(cards).toHaveLength(1);
    expect(cards[0].queryId).toBe("only");
  });
});
