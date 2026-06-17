"use client"; // kept for token-class posture; the data is static (build-imported curation, no fetch).

import Link from "next/link";
import { buildCards, groupByShape, displayShape } from "@/lib/gallery";
import GalleryCard from "@/components/GalleryCard";

// The gallery front door: a static, credential-less surface that shows the system's BREADTH. It
// reads the build-imported curation directly (no fetchCatalog, no useEffect, no loading/API-down
// branch), groups the canned runs by question shape, and links each card INTO the analyst at
// `/?query=<id>` (the already-wired self-play load path). Responsive to a single column below 1024px.
export default function Gallery() {
  const sections = groupByShape(buildCards());

  return (
    <main className="gallery">
      <header className="bar">
        <div className="eyebrow"><span className="dot" /> Glass-Box Video Analyst · Runs Gallery</div>
        <div className="gx-head-row">
          <h1 className="gx-title">Every question shape this analyst answers</h1>
          <Link className="gx-back" href="/">← Back to analyst</Link>
        </div>
        <p className="gx-lede">
          Each card is a canned run grouped by question shape. Click one to watch the analyst
          generate a visual program and run it over the clip, step by step.
        </p>
      </header>

      {sections.map((s) => (
        <section key={s.shape} className="gx-section" aria-label={s.shape}>
          <h2 className="gx-shape">{displayShape(s.shape)}</h2>
          <div className="gx-grid">
            {s.cards.map((c) => <GalleryCard key={c.queryId} card={c} />)}
          </div>
        </section>
      ))}
    </main>
  );
}
