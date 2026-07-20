// Clip-aware evidence stills, sourced from the build-imported frames-manifest (no runtime fetch —
// mirrors gallery.ts's static-curation posture). The precompute pipeline overwrites
// frames-manifest.json with more/different clips and ts, but the shape (this file's contract)
// stays stable: { width, height, stills, still_path, poster_ts, video } per clip id.
//
// Root-cause fix for the customer-visible bug: EvidencePanel used to hardcode the hero clip's two
// stills by raw ts, so a bernabeu-counter run showed the hero's frames (or nothing). Every still
// lookup now goes through a clip id, so a step's frame can only ever come from ITS clip.

import type { ClipFrames, FramesManifest } from "./types";
import manifest from "./frames-manifest.json";

const MANIFEST = manifest as FramesManifest;

/** The manifest entry for a clip id, or null when the clip has no precomputed frames. */
export function clipFrames(clipId: string): ClipFrames | null {
  return MANIFEST[clipId] ?? null;
}

/** The nearest sampled ts to `ts` in an ascending `stills` list. Ties resolve to the earlier
 * (smaller) ts. null on an empty list. */
export function nearestTs(stills: number[], ts: number): number | null {
  if (stills.length === 0) return null;
  let best = stills[0];
  let bestDist = Math.abs(ts - best);
  for (let i = 1; i < stills.length; i++) {
    const dist = Math.abs(ts - stills[i]);
    if (dist < bestDist) {
      best = stills[i];
      bestDist = dist;
    }
    // dist === bestDist -> a tie against a LATER candidate; keep the earlier `best` (no update).
  }
  return best;
}

/** Fill a `still_path` template's single `{ts}` placeholder. */
function pathFor(entry: ClipFrames, ts: number): string {
  return entry.still_path.replace("{ts}", String(ts));
}

/**
 * The still URL for the nearest sampled ts in `clipId`'s manifest entry. null when ts is 0/absent
 * (no meaningful frame — e.g. an ungrounded answer step) or the clip is unknown to the manifest.
 * Pure; used by EvidencePanel so a step's frame can only ever come from its own clip.
 */
export function frameSrc(clipId: string | null | undefined, ts: number | null | undefined): string | null {
  if (!ts) return null;
  if (!clipId) return null;
  const entry = clipFrames(clipId);
  if (!entry) return null;
  const nearest = nearestTs(entry.stills, ts);
  if (nearest == null) return null;
  return pathFor(entry, nearest);
}

/** The clip's poster still (gallery thumbnail / <video poster>). null when the clip is unknown. */
export function posterSrc(clipId: string): string | null {
  const entry = clipFrames(clipId);
  return entry ? pathFor(entry, entry.poster_ts) : null;
}
