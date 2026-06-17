import type { GalleryCardVM } from "@/lib/types";
import { displayShape } from "@/lib/gallery";

// One gallery card: a clip still (or a token placeholder for clips with no committed still) above
// the query text (Fraunces headline-ish), a shape tag (mono, teal hairline pill) and the clip
// label (muted sans). NO confidence number and NO cached-answer preview (Resolved #2). Pure
// presentational — it consumes a typed view-model and never fetches. The whole card is an anchor
// to `/?query=<id>` so it self-plays that canned query (real navigation survives static export).
export default function GalleryCard({ card }: { card: GalleryCardVM }) {
  return (
    <a className="gx-card" href={card.href}>
      <div className="gx-thumb">
        {card.thumb.kind === "img" ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img className="gx-thumb-img" src={card.thumb.src} alt="" />
        ) : (
          <div className="gx-thumb-ph" aria-hidden>
            <span className="gx-ph-clip">{card.clipLabel}</span>
            <span className="gx-ph-shape">{displayShape(card.shape)}</span>
          </div>
        )}
      </div>
      <div className="gx-body">
        <h3 className="gx-query">{card.text}</h3>
        <div className="gx-meta">
          <span className="gx-shape-tag">{displayShape(card.shape)}</span>
          <span className="gx-clip-label">{card.clipLabel}</span>
        </div>
      </div>
    </a>
  );
}
