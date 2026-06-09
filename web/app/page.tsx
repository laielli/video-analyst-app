"use client";

import { useEffect, useRef, useState } from "react";
import { useRun, fetchCatalog, permalinkUrl } from "@/lib/useRun";
import type { CatalogQuery } from "@/lib/types";
import ProgramPanel from "@/components/ProgramPanel";
import EvidencePanel from "@/components/EvidencePanel";
import StepTracker from "@/components/StepTracker";
import Findings from "@/components/Findings";

/** Read `?run=` / `?query=` off the current URL (empty on the server / when unset). */
function urlParams(): { run: string | null; query: string | null } {
  if (typeof window === "undefined") return { run: null, query: null };
  const p = new URLSearchParams(window.location.search);
  return { run: p.get("run"), query: p.get("query") };
}

export default function Page() {
  const [queries, setQueries] = useState<CatalogQuery[]>([]);
  const [queryId, setQueryId] = useState<string | null>(null);
  const [text, setText] = useState(""); // the typed free-text question
  const [err, setErr] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const run = useRun(queryId, 1200);
  // A `?run=<id>` permalink on load replays a stored run; gate the canned auto-play so it doesn't
  // race and clobber the replay. Captured once at mount (the URL doesn't change after).
  const permalinkRef = useRef<string | null>(null);
  if (permalinkRef.current === null) permalinkRef.current = urlParams().run ?? "";

  useEffect(() => {
    const { run: runId, query: queryParam } = urlParams();
    // `?run=<id>` on load: replay the shared stored run immediately (bypasses the catalog).
    if (runId) run.run({ kind: "permalink", runId });
    fetchCatalog()
      .then((c) => {
        setQueries(c.queries);
        // `?query=<id>` selects that canned query on load; else default to the first.
        const wanted = queryParam && c.queries.some((q) => q.id === queryParam)
          ? queryParam
          : c.queries[0]?.id ?? null;
        if (wanted) setQueryId(wanted);
      })
      .catch(() => setErr("Can't reach the API. Start it: cd api && uvicorn server:app --port 8000"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-run on first load and whenever the canned query changes (the public URL self-plays —
  // D-DR7). Free-text runs are explicit (the Ask button), never auto-run on keystroke. GATED on
  // no `?run=` permalink present, or it would race and clobber the permalink replay.
  useEffect(() => {
    if (!queryId) return;
    if (permalinkRef.current) return; // a permalink replay owns this load
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
    run.run({ kind: "free", text: t, clip: clipId });
  };

  // Copy a shareable permalink to the current grounded free-text run. Enabled only when the
  // streamed meta carries a run_id (grounded free-text run or a permalink replay).
  const runId = run.meta?.run_id;
  const copyRunLink = async () => {
    if (!runId || typeof window === "undefined") return;
    try {
      await navigator.clipboard.writeText(permalinkUrl(window.location.origin, runId));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard denied — no-op (the link is still derivable from the URL). */
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
          <div className="eyebrow"><span className="dot" /> Glass-Box Video Analyst · Portfolio Demo</div>
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
                >
                  {copied ? "Copied!" : "Copy run link"}
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
