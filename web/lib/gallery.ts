import type { CurationEntry, GalleryCard, GallerySection, GalleryThumb } from "./types";
import curated from "./curated.json"; // build-imported static curation; NO runtime fetch.

// Pure helpers for the gallery front door. No DOM, no network — mirrors web/lib/program.ts's
// posture so the whole module is unit-testable. The gallery's data source is the build-imported
// static curation above; drift off canned.QUERIES + the eval bank is neutralized by the backend
// parity test (api/tests/test_gallery_curation_parity.py).

// Deterministic section order over the shapes that actually exist in the canned set. The set has
// NO presence query, so "presence" is intentionally absent — groupByShape suppresses empty groups
// rather than emitting a hollow section. Unknown shapes sort last (stable, defensive).
export const SHAPE_ORDER = ["count", "first-goal", "scorer-number", "scene"] as const;

// machine shape -> display label, mapped ONLY at render time (never minted in the data layer).
const SHAPE_DISPLAY: Record<string, string> = {
  count: "Count",
  "first-goal": "First-goal",
  "scorer-number": "Scorer-number",
  scene: "Scene",
};

// Pinned still assignment (Phase 2 / R1): only two stills exist, both from `single-goal`. The two
// single-goal cards each get one; every bernabeu-counter card falls back to a token placeholder.
const STILL_BY_QUERY: Record<string, string> = {
  "hero-10-first-goal": "/frames/goal-4000.jpg",
  "single-goal-scene": "/frames/scorer-4625.jpg",
};

/** The entry's lowercase machine shape (single source of truth lives in curated.json). */
export function shapeFor(entry: CurationEntry): string {
  return entry.shape;
}

/** machine shape -> display text (Title-ish). Falls back to the raw machine label if unmapped. */
export function displayShape(shape: string): string {
  return SHAPE_DISPLAY[shape] ?? shape;
}

/** The entry's human clip label; falls back to the raw clip id if absent (defensive). */
export function clipLabel(entry: CurationEntry): string {
  return entry.clipLabel ?? entry.clip;
}

/** A committed still when one is pinned for this query, else a token-built placeholder. */
export function thumbFor(entry: CurationEntry): GalleryThumb {
  const src = STILL_BY_QUERY[entry.id];
  return src ? { kind: "img", src } : { kind: "placeholder" };
}

/** The analyst load path: "/?query=<id>" (URL-encoded). The analyst self-plays that canned run. */
export function linkFor(entry: CurationEntry): string {
  return `/?query=${encodeURIComponent(entry.id)}`;
}

/** Assemble one view-model card per curated entry (no fetch — the curation is build-imported). */
export function buildCards(curation: CurationEntry[] = curated): GalleryCard[] {
  return curation.map((q) => ({
    queryId: q.id,
    clipId: q.clip,
    clipLabel: clipLabel(q),
    text: q.text,
    shape: shapeFor(q),
    href: linkFor(q),
    thumb: thumbFor(q),
  }));
}

/** Group cards by machine shape in deterministic SHAPE_ORDER; suppress empty groups; unknown
 * shapes sort last (stable). */
export function groupByShape(cards: GalleryCard[]): GallerySection[] {
  const byShape = new Map<string, GalleryCard[]>();
  for (const c of cards) {
    (byShape.get(c.shape) ?? byShape.set(c.shape, []).get(c.shape)!).push(c);
  }
  const rank = (shape: string) => {
    const i = SHAPE_ORDER.indexOf(shape as (typeof SHAPE_ORDER)[number]);
    return i === -1 ? SHAPE_ORDER.length : i; // unknown shapes sort last
  };
  return [...byShape.keys()]
    .sort((a, b) => rank(a) - rank(b))
    .map((shape) => ({ shape, cards: byShape.get(shape)! }));
}
