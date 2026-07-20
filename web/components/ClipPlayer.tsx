"use client";

import { useRef } from "react";
import { clipFrames, posterSrc } from "@/lib/frames";

/**
 * The reviewer's top-complaint fix: "I should be able to view the original clip." A plain,
 * a11y-quiet <video> for the analyzed clip — no autoplay, controls always visible, poster from
 * the clip's manifest still. `focusTs` (a step's evidence.frame_ts_ms, source-timeline) drives an
 * optional "jump to this frame" affordance; the clip's own video file starts at `offset_ms` on
 * that same source timeline (bernabeu's video starts 7870ms in), so the seek target is
 * `(focusTs - offset_ms) / 1000` seconds INTO the video file, not the raw source ts.
 */
export default function ClipPlayer({
  clipId, clipLabel, focusTs,
}: { clipId: string | null | undefined; clipLabel?: string; focusTs?: number | null }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const entry = clipId ? clipFrames(clipId) : null;

  // No clip selected yet (idle, no catalog clip) or a clip the manifest doesn't know about —
  // nothing playable to show. EvidencePanel's empty state covers the "no data" story elsewhere.
  if (!clipId || !entry) return null;

  const poster = posterSrc(clipId) ?? undefined;
  const label = clipLabel || clipId;
  const canJump = focusTs != null && focusTs > 0;

  const jumpToFrame = () => {
    const v = videoRef.current;
    if (!v || focusTs == null) return;
    v.currentTime = Math.max(0, (focusTs - entry.video.offset_ms) / 1000);
  };

  return (
    <section className="panel clip-player" aria-label={`Original clip: ${label}`}>
      <div className="panel-head">
        <div className="panel-title">Clip <span className="sub">{label}</span></div>
        {canJump && (
          <button
            type="button"
            className="btn cp-jump"
            onClick={jumpToFrame}
            aria-label="Jump to this step's frame in the clip"
          >
            Jump to this frame
          </button>
        )}
      </div>
      <div className="cp-body">
        {/* No autoplay (reduced-motion / a11y): the reviewer opts in by pressing play. */}
        <video
          ref={videoRef}
          className="cp-video"
          src={entry.video.src}
          poster={poster}
          controls
          muted
          playsInline
          preload="metadata"
          aria-label={`Video: ${label}`}
        />
      </div>
    </section>
  );
}
