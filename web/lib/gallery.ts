// Pure helpers for the gallery / front-door surface. No DOM, no network — the data source is the
// build-imported static curation (web/lib/curated.json), so the gallery renders with ZERO runtime
// API dependency (a credential-less / API-down front door still shows the system's breadth). The
// curation can NEVER drift off canned.QUERIES / canned.CLIPS / the eval bank — that is enforced by
// api/tests/test_gallery_curation_parity.py. Mirrors web/lib/program.ts's pure-lib posture.

import type {
  CuratedEntry,
  GalleryCardVM,
  GallerySection,
  GalleryThumb,
} from "./types";
import curated from "./curated.json";

// The committed stills are BOTH from the single-goal clip (verified: only goal-4000.jpg and
// scorer-4625.jpg exist under web/public/frames). bernabeu-counter has no committed still, so its
// cards fall back to a token-driven CSS placeholder. Pinned still assignment keeps thumbnails
// deterministic (test #5).
const STILL_BY_QUERY: Record<string, string> = {
  "hero-10-first-goal": "/frames/goal-4000.jpg",
  "single-goal-scene": "/frames/scorer-4625.jpg",
};

// Deterministic section order over the shapes that ACTUALLY exist in the canned set. The canned
// set has no `presence` query, so SHAPE_ORDER must not emit a hollow "Presence" section — empty
// groups are suppressed by groupByShape. Unknown shapes sort last (stable).
export const SHAPE_ORDER = ["count", "first-goal", "scorer-number", "scene"] as const;

// Machine shape label -> display text. Mapping happens ONLY at render time (the data layer never
// mints title-cased labels). Unknown shapes pass through unchanged.
const SHAPE_DISPLAY: Record<string, string> = {
  count: "Count",
  "first-goal": "First-goal",
  "scorer-number": "Scorer-number",
  scene: "Scene",
};

/** Display text for a machine shape label (render-time only). Unknown shapes pass through. */
export function displayShape(shape: string): string {
  return SHAPE_DISPLAY[shape] ?? shape;
}

/** The entry's lowercase machine shape label. Map machine->display only at render time. */
export function shapeFor(entry: CuratedEntry): string {
  return entry.shape;
}

/** The entry's human clip label; falls back to the raw clip id if missing (defensive). */
export function clipLabel(entry: CuratedEntry): string {
  return entry.clipLabel || entry.clip;
}

/** Thumbnail: a committed still when one exists for this query, else a token CSS placeholder. */
export function thumbFor(entry: CuratedEntry): GalleryThumb {
  const src = STILL_BY_QUERY[entry.id];
  return src ? { kind: "img", src } : { kind: "placeholder" };
}

/** The link into the analyst's canned load path — exactly `/?query=<id>`, URL-encoded. */
export function hrefFor(entry: CuratedEntry): string {
  return `/?query=${encodeURIComponent(entry.id)}`;
}

/** Build one view-model per curated entry. Defaults to the build-imported curation (no fetch). */
export function buildCards(curation: CuratedEntry[] = curated): GalleryCardVM[] {
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
 * Group cards by machine shape in deterministic SHAPE_ORDER (unknown shapes last, preserving
 * first-seen order). Empty groups are suppressed — only shapes with at least one card appear.
 */
export function groupByShape(cards: GalleryCardVM[]): GallerySection[] {
  const byShape = new Map<string, GalleryCardVM[]>();
  for (const card of cards) {
    const bucket = byShape.get(card.shape);
    if (bucket) bucket.push(card);
    else byShape.set(card.shape, [card]);
  }

  const rank = (shape: string): number => {
    const i = (SHAPE_ORDER as readonly string[]).indexOf(shape);
    return i === -1 ? SHAPE_ORDER.length : i;
  };

  return [...byShape.keys()]
    .sort((a, b) => rank(a) - rank(b))
    .map((shape) => ({ shape, cards: byShape.get(shape)! }));
}
