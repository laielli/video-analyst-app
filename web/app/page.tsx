"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRun, fetchCatalog, shareLink } from "@/lib/useRun";
import type { CatalogQuery } from "@/lib/types";
import ProgramPanel from "@/components/ProgramPanel";
import EvidencePanel from "@/components/EvidencePanel";
import StepTracker from "@/components/StepTracker";
import Findings from "@/components/Findings";

/** Read `?run=<id>` and `?query=<id>` from the current URL (empty strings -> null). SSR-safe. */
function urlParams(): { run: string | null; query: string | null } {
  if (typeof window === "undefined") return { run: null, query: null };
  const p = new URLSearchParams(window.location.search);
  return { run: p.get("run") || null, query: p.get("query") || null };
}

export default function Page() {
  const [queries, setQueries] = useState<CatalogQuery[]>([]);
  const [queryId, setQueryId] = useState<string | null>(null);
  const [text, setText] = useState(""); // the typed free-text question
  const [err, setErr] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const run = useRun(queryId, 1200);

  useEffect(() => {
    fetchCatalog()
      .then((c) => {
        setQueries(c.queries);
        // `?query=<id>` selects that canned query on load; else default to the first.
        const { query: queryParam } = urlParams();
        const wanted = queryParam && c.queries.some((q) => q.id === queryParam) ? queryParam : null;
        const next = wanted ?? c.queries[0]?.id ?? null;
        if (next) setQueryId(next);
      })
      .catch(() => setErr("Can't reach the API. Start it: cd api && uvicorn server:app --port 8000"));
  }, []);

  // On mount, a `?run=<id>` permalink auto-replays the stored run (highest priority — the
  // share-link entry point). It must run BEFORE/INSTEAD OF the canned auto-play, which is why the
  // canned effect below is gated on the absence of a `?run=` param.
  useEffect(() => {
    const { run: runId } = urlParams();
    if (runId) run.run({ kind: "permalink", runId });
    return () => run.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-run on first load and whenever the canned query changes (the public URL self-plays —
  // D-DR7). Free-text runs are explicit (the Ask button), never auto-run on keystroke. GATED so a
  // `?run=<id>` permalink replay is not clobbered: this effect fires the moment fetchCatalog
  // resolves, so without the guard it would race the permalink replay above.
  useEffect(() => {
    if (!queryId) return;
    if (urlParams().run) return; // a permalink replay owns the first run — don't clobber it.
    run.run();
    return () => run.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queryId]);

  const selected = queries.find((q) => q.id === queryId);
  // v1 has a single clip; carry its id so a free-text question runs against the right clip.
  const clipId = selected?.clip;

  // Submit the typed question (no-op on empty/whitespace). The streamed meta.query then reflects
  // the user's words, so the heading shows the typed question rather than the canned label.
  const ask = () => {
    const t = text.trim();
    if (!t) return;
    setCopied(false);
    run.run({ kind: "free", text: t, clip: clipId });
  };

  // "Copy run link": enabled only when the current run minted a run_id (a grounded free-text run
  // OR a permalink replay). Writes `${origin}/?run=<id>` to the clipboard so the run is shareable.
  const runId = run.meta?.run_id;
  const copyRunLink = async () => {
    if (!runId || typeof window === "undefined") return;
    try {
      await navigator.clipboard.writeText(shareLink(window.location.origin, runId));
      setCopied(true);
    } catch {
      // clipboard blocked (e.g. insecure context) — leave the button label unchanged.
    }
  };

  const program = run.meta?.program ?? [];
  const currentStep = run.current >= 0 ? run.steps[run.current] ?? null : null;
  const total = run.meta?.total_steps ?? program.length;
  // The heading reflects the live run's query when present (free text), else the canned label.
  const heading = run.meta?.query ?? selected?.text ?? "Loading…";

  return (
    <>
      <div className="gate">
        <div>
          <h1>Best viewed on desktop</h1>
          <p>The Glass-Box Video Analyst is a two-panel code + evidence tool. Open it on a wider screen (≥1024px).</p>
        </div>
      </div>

      <main className="app">
        <header className="bar">
          <div className="eyebrow">
            <span className="dot" /> Glass-Box Video Analyst · Portfolio Demo
            <Link className="btn gx-browse" href="/gallery" style={{ marginLeft: "auto" }}>Browse all runs →</Link>
          </div>
          <div className="query-row">
            <h1 className="query">{heading}</h1>
            <div className="controls">
              {queries.length > 1 && (
                <select className="select" value={queryId ?? ""} onChange={(e) => setQueryId(e.target.value)}>
                  {queries.map((q) => <option key={q.id} value={q.id}>{q.text}</option>)}
                </select>
              )}
              <button className="btn btn-run" onClick={() => run.run()} disabled={run.status === "running"}>
                {run.status === "running" ? "Running…" : run.status === "done" ? "Replay" : "Run"}
              </button>
              {runId && (
                <button
                  className="btn btn-share"
                  onClick={copyRunLink}
                  aria-label="Copy a shareable link to this run"
                  title="Copy a shareable link to this run"
                >
                  {copied ? "Link copied" : "Copy run link"}
                </button>
              )}
            </div>
          </div>
          {/* Free-text query: type a question, click Ask. The interpreter runs over this clip's
              precomputed cache, so most novel questions honestly "couldn't ground" (by design). */}
          <form className="ask-row" onSubmit={(e) => { e.preventDefault(); ask(); }}>
            <input
              className="ask-input"
              type="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder={selected ? `Ask about this clip… e.g. "${selected.text}"` : "Ask about this clip…"}
              aria-label="Ask a free-text question about the clip"
            />
            <button className="btn ask-btn" type="submit" disabled={run.status === "running" || !text.trim()}>
              Ask
            </button>
            {clipId && <span className="clip-label" aria-label="Clip being analyzed">clip: {clipId}</span>}
          </form>
        </header>

        {err && <p style={{ color: "var(--amber)", marginTop: 16, fontFamily: "var(--ff-mono)", fontSize: 13 }}>{err}</p>}

        <div className="panels">
          <ProgramPanel program={program} steps={run.steps} current={run.current} />
          <EvidencePanel step={currentStep} programSource={run.meta?.program_source} />
        </div>

        <section className="tracker" aria-label="Execution pipeline">
          <div className="tracker-top">
            <Findings status={run.status} findings={run.findings} stepCount={run.steps.length} total={total} />
            <div className="legend" aria-hidden>
              <span className="it"><span className="sw done" /> Completed</span>
              <span className="it"><span className="sw active" /> Active</span>
              <span className="it"><span className="sw pending" /> Pending</span>
            </div>
          </div>
          <StepTracker program={program} steps={run.steps} current={run.current} onSelect={run.select} />
        </section>
      </main>
    </>
  );
}
