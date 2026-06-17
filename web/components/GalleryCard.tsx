import type { GalleryCard as GalleryCardVM } from "@/lib/types";
import { displayShape } from "@/lib/gallery";

// One gallery card: a thumbnail (committed still OR token-built placeholder), the query text
// (Fraunces), a shape tag (mono, teal hairline pill) + clip label (muted sans). NO confidence
// number, NO cached-answer preview (Resolved #2 / DESIGN-ROOM D). Pure presentational — consumes a
// fully-resolved view-model and renders a real <a> so navigation survives static export.
export default function GalleryCard({ card }: { card: GalleryCardVM }) {
  return (
    <a className="gx-card" href={card.href} aria-label={`Run: ${card.text}`}>
      <div className="gx-thumb">
        {card.thumb.kind === "img" ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img className="gx-img" src={card.thumb.src} alt={`Still from ${card.clipLabel}`} />
        ) : (
          // Token-built placeholder for a clip with no committed still (mirrors .frame-empty).
          <div className="gx-placeholder" aria-label={`No still for ${card.clipLabel}`}>
            <span className="gx-ph-label">{card.clipLabel}</span>
            <span className="gx-ph-shape">{displayShape(card.shape)}</span>
          </div>
        )}
      </div>
      <div className="gx-body">
        <h3 className="gx-text">{card.text}</h3>
        <div className="gx-tags">
          <span className="gx-shape-tag">{displayShape(card.shape)}</span>
          <span className="gx-clip-label">{card.clipLabel}</span>
        </div>
      </div>
    </a>
  );
}
