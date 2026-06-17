import type { CurationEntry, GalleryCard, GallerySection, GalleryThumb } from "./types";
import curated from "./curated.json"; // build-imported static curation; NO runtime fetch (Resolved #4).

// Pure helpers for the gallery front door. Mirrors web/lib/program.ts posture: no DOM, no network,
// fully unit-testable. The data source is the build-imported curation above; the gallery route never
// calls fetchCatalog. Drift off canned.QUERIES / the eval bank is caught by the backend parity test.

// Deterministic section order over the shapes that actually exist in the canned set. The canned set
// has NO presence query, so "presence" is deliberately absent (no hollow section). Unknown shapes
// sort last (stable among themselves via first-seen order).
export const SHAPE_ORDER = ["count", "first-goal", "scorer-number", "scene"] as const;

// Machine shape label -> display text rendered in the UI. Kept here (the lib) so the data layer
// never mints title-cased labels; the route reads it via displayShape().
const SHAPE_DISPLAY: Record<string, string> = {
  count: "Count",
  "first-goal": "First-goal",
  "scorer-number": "Scorer-number",
  scene: "Scene",
};

// Committed still assignment (pinned so tests are deterministic — plan Phase 2 / R1). Only the two
// single-goal stills exist in the repo; every other entry falls back to the token placeholder.
const STILL_FOR: Record<string, string> = {
  "hero-10-first-goal": "/frames/goal-4000.jpg",
  "single-goal-scene": "/frames/scorer-4625.jpg",
};

/** The entry's lowercase machine shape (count / first-goal / scorer-number / scene). */
export function shapeFor(entry: CurationEntry): string {
  return entry.shape;
}

/** Map a machine shape label to its display string (e.g. count -> "Count"). Falls back to the raw
 * machine label for an unknown shape so the UI never renders blank. */
export function displayShape(shape: string): string {
  return SHAPE_DISPLAY[shape] ?? shape;
}

/** The entry's human clip label; falls back to the raw clip id when missing (defensive). */
export function clipLabel(entry: CurationEntry): string {
  return entry.clipLabel || entry.clip;
}

/** A committed still for entries that have one, else a token-built CSS placeholder. Exercises both
 * sides of the branch in test #5. */
export function thumbFor(entry: CurationEntry): GalleryThumb {
  const src = STILL_FOR[entry.id];
  return src ? { kind: "img", src } : { kind: "placeholder" };
}

/** Build one view-model per curated entry. Defaults to the build-imported curation. */
export function buildCards(curation: CurationEntry[] = curated as CurationEntry[]): GalleryCard[] {
  return curation.map((q) => ({
    queryId: q.id,
    clipId: q.clip,
    clipLabel: clipLabel(q),
    text: q.text,
    shape: shapeFor(q),
    href: `/?query=${encodeURIComponent(q.id)}`,
    thumb: thumbFor(q),
  }));
}

/** Group cards by shape in deterministic SHAPE_ORDER (unknown shapes last, first-seen order).
 * Empty groups are suppressed — a shape in SHAPE_ORDER with no cards emits no section. */
export function groupByShape(cards: GalleryCard[]): GallerySection[] {
  const byShape = new Map<string, GalleryCard[]>();
  for (const c of cards) {
    (byShape.get(c.shape) ?? byShape.set(c.shape, []).get(c.shape)!).push(c);
  }
  const rank = (shape: string) => {
    const i = (SHAPE_ORDER as readonly string[]).indexOf(shape);
    return i === -1 ? SHAPE_ORDER.length : i;
  };
  return [...byShape.keys()]
    .sort((a, b) => rank(a) - rank(b))
    .map((shape) => ({ shape, cards: byShape.get(shape)! }));
}
