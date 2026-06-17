"use client"; // kept for token-class posture; the data is static (build-imported curation).

import { buildCards, groupByShape, displayShape } from "@/lib/gallery";
import GalleryCard from "@/components/GalleryCard";

// The gallery / front door. Reads the build-imported static curation directly — NO fetchCatalog,
// NO useEffect, NO loading state, NO API-down branch. So a credential-less / API-down / not-yet-
// deployed-backend front door still shows the system's full breadth. Each card links into the
// analyst at `/?query=<id>`, which self-plays that canned query.
export default function Gallery() {
  const sections = groupByShape(buildCards());

  return (
    <main className="gallery">
      <header className="bar gx-bar">
        <div className="eyebrow"><span className="dot" /> Glass-Box Video Analyst · Run Gallery</div>
        <div className="gx-head">
          <h1 className="gx-title">Browse every run</h1>
          {/* Real <a> navigation (not next/link) to keep parity with the card links and
              survive static export; the repo uses no next/link. Same intentional-override
              posture as the <img> lint-disable in EvidencePanel.tsx. */}
          {/* eslint-disable-next-line @next/next/no-html-link-for-pages */}
          <a className="gx-back" href="/">← Open the analyst</a>
        </div>
        <p className="gx-lede">
          The analyst answers a fixed catalog of questions about two clips — counts, first-goal
          checks, a scorer readout, a whole-scene description. Pick any run to watch the generated
          program execute over the video, step by step.
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
