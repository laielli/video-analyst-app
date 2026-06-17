import type { GalleryCardVM } from "@/lib/types";
import { displayShape } from "@/lib/gallery";

// One gallery card: a real anchor into the analyst (/?query=<id>) wrapping a thumbnail + query
// text (Fraunces) + a shape tag (mono, teal hairline pill) + clip label (muted sans). NO
// confidence number, NO cached-answer preview (Resolved #2 = DESIGN-ROOM D). Pure presentational —
// consumes a typed view-model. All visuals come from globals.css CSS vars (no inline hex/fonts).
export default function GalleryCard({ card }: { card: GalleryCardVM }) {
  return (
    <a className="gx-card" href={card.href}>
      <div className="gx-thumb">
        {card.thumb.kind === "img" ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={card.thumb.src} alt={`Still from ${card.clipLabel}`} />
        ) : (
          // Token-built placeholder for clips without a committed still (mirrors .frame-empty).
          <div className="gx-thumb-empty" aria-label="No still for this clip">
            <span className="gx-thumb-clip">{card.clipLabel}</span>
            <span className="gx-thumb-shape">{displayShape(card.shape)}</span>
          </div>
        )}
      </div>
      <div className="gx-card-body">
        <p className="gx-query">{card.text}</p>
        <div className="gx-tags">
          <span className="gx-shape-tag">{displayShape(card.shape)}</span>
          <span className="gx-clip-label">{card.clipLabel}</span>
        </div>
      </div>
    </a>
  );
}
