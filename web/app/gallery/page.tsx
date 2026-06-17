"use client"; // kept for token-class posture; the data is static (build-imported curation).

import Link from "next/link";
import { buildCards, groupByShape, displayShape } from "@/lib/gallery";
import GalleryCard from "@/components/GalleryCard";

// The gallery / demo front door. Reads the build-imported static curation directly — NO
// fetchCatalog, NO useEffect, NO loading/API-down branch — so a credential-less / API-down /
// not-yet-deployed-backend front door still shows the system's full breadth. Cards group by
// question shape and link INTO the analyst at /?query=<id>, which self-plays that canned run.

export default function Gallery() {
  const sections = groupByShape(buildCards());

  return (
    <main className="gallery">
      <header className="bar">
        <div className="eyebrow"><span className="dot" /> Glass-Box Video Analyst · Run Gallery</div>
        <div className="query-row">
          <h1 className="query">Browse every run the analyst answers</h1>
          <div className="controls">
            <Link className="btn" href="/">← Back to analyst</Link>
          </div>
        </div>
        <p className="gx-lede">
          Each card is a canned run grouped by question shape. Open one to watch the analyst
          generate a visual program and run it over the clip, step by step.
        </p>
      </header>

      {sections.map((s) => (
        <section key={s.shape} className="gx-section" aria-label={displayShape(s.shape)}>
          <h2 className="gx-shape">{displayShape(s.shape)}</h2>
          <div className="gx-grid">
            {s.cards.map((c) => <GalleryCard key={c.queryId} card={c} />)}
          </div>
        </section>
      ))}
    </main>
  );
}
