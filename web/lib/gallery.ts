import type {
  CurationEntry,
  GalleryCardVM,
  GallerySection,
  GalleryThumb,
} from "./types";
import curated from "./curated.json"; // build-imported (resolveJsonModule); NO runtime fetch.

// Pure helpers for the gallery front-door. No DOM, no network — the data source is the static
// curation bundled at curated.json (Resolved #4). Mirrors web/lib/program.ts posture: deterministic,
// unit-testable. The single source of truth (canned.QUERIES + the eval bank) is enforced by the
// backend parity test api/tests/test_gallery_curation_parity.py.

// Deterministic section order over the shapes that actually exist in the canned set. The set has
// NO presence query, so "presence" is intentionally absent — groupByShape suppresses empty groups,
// so we never emit a hollow section even if SHAPE_ORDER listed one. Unknown shapes sort last.
export const SHAPE_ORDER = ["count", "first-goal", "scorer-number", "scene"];

// Machine shape label -> display text (title-cased). Mapping lives in the lib so both the route
// and tests share one source; data layer never mints title-cased labels (Resolved B).
const SHAPE_DISPLAY: Record<string, string> = {
  count: "Count",
  "first-goal": "First-goal",
  "scorer-number": "Scorer-number",
  scene: "Scene",
};

// R1: which committed still each single-goal card uses. bernabeu-counter cards have NO committed
// still and fall through to the token placeholder. Pinned so the card thumbnail is deterministic.
const STILL_FOR: Record<string, string> = {
  "hero-10-first-goal": "/frames/goal-4000.jpg",
  "single-goal-scene": "/frames/scorer-4625.jpg",
};

/** The entry's lowercase machine shape label. Map machine -> display only at render time. */
export function shapeFor(entry: CurationEntry): string {
  return entry.shape;
}

/** Display text for a machine shape; falls back to the raw machine label for an unknown shape. */
export function displayShape(shape: string): string {
  return SHAPE_DISPLAY[shape] ?? shape;
}

/** The entry's human clip label; falls back to the raw clip id when absent (defensive). */
export function clipLabel(entry: CurationEntry): string {
  return entry.clipLabel || entry.clip;
}

/** R1 thumbnail: a committed still for pinned single-goal cards, else the token placeholder. */
export function thumbFor(entry: CurationEntry): GalleryThumb {
  const src = STILL_FOR[entry.id];
  return src ? { kind: "img", src } : { kind: "placeholder" };
}

/** Build the per-entry view-model (card-per-query). Link target is the analyst at /?query=<id>. */
export function buildCards(curation: CurationEntry[] = curated): GalleryCardVM[] {
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

/** Group cards by shape in deterministic SHAPE_ORDER; suppress empty groups; unknown shapes last. */
export function groupByShape(cards: GalleryCardVM[]): GallerySection[] {
  const byShape = new Map<string, GalleryCardVM[]>();
  for (const c of cards) {
    (byShape.get(c.shape) ?? byShape.set(c.shape, []).get(c.shape)!).push(c);
  }
  const rank = (shape: string) => {
    const i = SHAPE_ORDER.indexOf(shape);
    return i === -1 ? SHAPE_ORDER.length : i; // unknown shapes sort last
  };
  return [...byShape.keys()]
    .sort((a, b) => rank(a) - rank(b) || a.localeCompare(b))
    .map((shape) => ({ shape, cards: byShape.get(shape)! }));
}
