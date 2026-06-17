import { displayShape } from "@/lib/gallery";
import type { GalleryCardVM } from "@/lib/types";

// One gallery card: a real anchor (survives static export) into the analyst at /?query=<id>,
// wrapping a token-driven thumbnail + the query text (Fraunces, via .gx-text) + a shape tag (mono
// teal hairline pill) + the clip label (muted sans). Pure presentational — no fetch, no preview,
// no confidence (Resolved #2/#D). All visuals come from globals.css CSS vars.

export default function GalleryCard({ card }: { card: GalleryCardVM }) {
  return (
    <a className="gx-card" href={card.href} aria-label={`Open analyst: ${card.text}`}>
      <div className="gx-thumb">
        {card.thumb.kind === "img" ? (
          // Follows the EvidencePanel precedent for the static export <img> (no next/image).
          // eslint-disable-next-line @next/next/no-img-element
          <img src={card.thumb.src} alt="" className="gx-thumb-img" />
        ) : (
          <div className="gx-thumb-placeholder" aria-hidden>
            <span className="gx-thumb-clip">{card.clipLabel}</span>
            <span className="gx-thumb-shape">{displayShape(card.shape)}</span>
          </div>
        )}
      </div>
      <div className="gx-body">
        <h3 className="gx-text">{card.text}</h3>
        <div className="gx-meta">
          <span className="gx-shape-tag">{displayShape(card.shape)}</span>
          <span className="gx-clip">{card.clipLabel}</span>
        </div>
      </div>
    </a>
  );
}
