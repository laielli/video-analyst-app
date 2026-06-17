"use client"; // kept for token-class posture; the data is static (build-imported curation).

import Link from "next/link";
import { buildCards, groupByShape, displayShape } from "@/lib/gallery";
import GalleryCard from "@/components/GalleryCard";

// The runs gallery / demo front door. Reads the build-imported static curation directly — NO
// fetchCatalog, NO useEffect, NO loading state, NO API-down branch (Resolved #4). Renders the full
// canned catalog as browsable cards grouped by question shape; each card links INTO the analyst at
// /?query=<id>, which self-plays that canned run. Zero runtime API dependency.
export default function Gallery() {
  const sections = groupByShape(buildCards());

  return (
    <main className="gallery">
      <header className="bar">
        <div className="eyebrow"><span className="dot" /> Glass-Box Video Analyst · Runs Gallery</div>
        <div className="query-row">
          <h1 className="query">Every question the analyst answers</h1>
          <Link className="btn" href="/">← Open the analyst</Link>
        </div>
        <p className="gallery-lede">
          Browse the canned runs by question shape. Click any card to open it in the analyst — it
          self-plays the program, frame by frame, with the evidence behind every step.
        </p>
      </header>

      {sections.map((s) => (
        <section key={s.shape} className="gx-section" aria-label={s.shape}>
          <h2 className="gx-shape">{displayShape(s.shape)}</h2>
          <div className="gx-grid">
            {s.cards.map((c) => (
              <GalleryCard key={c.queryId} card={c} />
            ))}
          </div>
        </section>
      ))}
    </main>
  );
}
