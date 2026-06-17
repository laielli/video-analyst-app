import type { GalleryCard as GalleryCardVM } from "@/lib/types";
import { displayShape } from "@/lib/gallery";

// One gallery card: a clip still (or token-built placeholder) above the query text (Fraunces),
// with a teal hairline shape pill (mono) + muted clip label (sans). Pure presentational —
// consumes a fully-resolved view-model. The whole card is an <a> so navigation survives the
// static export (real <a href>, no client router). NO confidence number, NO cached-answer
// preview (Resolved #2 = DESIGN-ROOM D).
export default function GalleryCard({ card }: { card: GalleryCardVM }) {
  return (
    <a className="gx-card" href={card.href} aria-label={`Run: ${card.text}`}>
      <div className="gx-thumb">
        {card.thumb.kind === "img" ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={card.thumb.src} alt={`${card.clipLabel} still`} />
        ) : (
          <div className="gx-thumb-ph" aria-label="No still for this clip">
            <span className="gx-ph-clip">{card.clipLabel}</span>
            <span className="gx-ph-shape">{displayShape(card.shape)}</span>
          </div>
        )}
      </div>
      <div className="gx-body">
        <h3 className="gx-q">{card.text}</h3>
        <div className="gx-tags">
          <span className="gx-shape-pill">{displayShape(card.shape)}</span>
          <span className="gx-clip">{card.clipLabel}</span>
        </div>
      </div>
    </a>
  );
}
