"use client"; // kept for token-class posture; the data is the build-imported static curation.

import Link from "next/link";
import { buildCards, groupByShape, displayShape } from "@/lib/gallery";
import GalleryCard from "@/components/GalleryCard";

// The demo front door. Reads the bundled static curation directly (no fetchCatalog, no useEffect,
// no loading state, no API-down branch — Resolved #4), renders cards grouped by question shape,
// each linking INTO the analyst at "/?query=<id>" (the already-wired self-play load path). So a
// credential-less / API-down / not-yet-deployed-backend front door still shows the system's range.
export default function Gallery() {
  const sections = groupByShape(buildCards());

  return (
    <main className="gallery">
      <header className="bar">
        <div className="eyebrow"><span className="dot" /> Glass-Box Video Analyst · Run Gallery</div>
        <div className="query-row">
          <h1 className="query">Every question the analyst answers</h1>
          <Link className="btn gx-back" href="/">← Open the analyst</Link>
        </div>
        <p className="gx-lede">
          A curated tour of the question shapes the visual-program runtime handles — counts, first-goal
          temporal reasoning, jersey-number readouts, whole-scene captions. Pick any run to watch it play.
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
