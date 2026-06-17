import type {
  CurationEntry,
  GalleryCardVM,
  GalleryThumb,
  GallerySection,
} from "./types";
import curated from "./curated.json"; // build-imported static curation; no runtime fetch.

// Pure gallery helpers (no DOM, no network) — mirrors web/lib/program.ts posture. The single data
// source is the build-imported curation; the gallery never touches /api/catalog (Resolved #4).

// Deterministic section order over the shapes that actually exist in the canned set. The canned
// set has NO presence query, so SHAPE_ORDER must not list one (groupByShape suppresses empties
// regardless, but listing only real shapes keeps intent obvious). Unknown shapes sort last.
export const SHAPE_ORDER = ["count", "first-goal", "scorer-number", "scene"] as const;

// Machine -> display label, applied only at render time (the data layer keeps machine labels).
const SHAPE_DISPLAY: Record<string, string> = {
  count: "Count",
  "first-goal": "First-goal",
  "scorer-number": "Scorer-number",
  scene: "Scene",
};

// Pinned still assignment so the thumbnail tests are deterministic. Only single-goal queries map
// to a committed still (both stills are single-goal frames); every other clip gets the token
// placeholder. Keyed by query id (not clip) so each card's still is explicit.
const STILL_BY_QUERY: Record<string, string> = {
  "hero-10-first-goal": "/frames/goal-4000.jpg",
  "single-goal-scene": "/frames/scorer-4625.jpg",
};

/** The lowercase machine shape for an entry. The UI maps machine -> display via displayShape. */
export function shapeFor(entry: CurationEntry): string {
  return entry.shape;
}

/** Title-cased display label for a machine shape (render-time only). Falls back to the raw label. */
export function displayShape(shape: string): string {
  return SHAPE_DISPLAY[shape] ?? shape;
}

/** The human clip label, falling back to the raw clip id when the entry omits it (defensive). */
export function clipLabel(entry: CurationEntry): string {
  return entry.clipLabel || entry.clip;
}

/** A committed still when one is pinned for this query, else a token CSS placeholder (per R1). */
export function thumbFor(entry: CurationEntry): GalleryThumb {
  const src = STILL_BY_QUERY[entry.id];
  return src ? { kind: "img", src } : { kind: "placeholder" };
}

/** The link INTO the analyst that self-plays this canned query: /?query=<id> (URL-encoded). */
export function hrefFor(entry: CurationEntry): string {
  return `/?query=${encodeURIComponent(entry.id)}`;
}

/** View-model per curated entry. Defaults to the build-imported curation. */
export function buildCards(curation: CurationEntry[] = curated as CurationEntry[]): GalleryCardVM[] {
  return curation.map((q) => ({
    queryId: q.id,
    clipId: q.clip,
    clipLabel: clipLabel(q),
    text: q.text,
    shape: shapeFor(q),
    href: hrefFor(q),
    thumb: thumbFor(q),
  }));
}

/**
 * Group cards by machine shape in deterministic order (SHAPE_ORDER first, unknown shapes after in
 * first-seen order). Empty groups are suppressed — the canned set has no presence query, so no
 * hollow section is emitted.
 */
export function groupByShape(cards: GalleryCardVM[]): GallerySection[] {
  const byShape = new Map<string, GalleryCardVM[]>();
  for (const card of cards) {
    (byShape.get(card.shape) ?? byShape.set(card.shape, []).get(card.shape)!).push(card);
  }
  const known = SHAPE_ORDER.filter((s) => byShape.has(s));
  const unknown = [...byShape.keys()].filter((s) => !SHAPE_ORDER.includes(s as (typeof SHAPE_ORDER)[number]));
  return [...known, ...unknown].map((shape) => ({ shape, cards: byShape.get(shape)! }));
}
