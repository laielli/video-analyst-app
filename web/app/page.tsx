"use client";

import { useEffect, useState } from "react";
import { useRun, fetchCatalog } from "@/lib/useRun";
import type { CatalogQuery } from "@/lib/types";
import ProgramPanel from "@/components/ProgramPanel";
import EvidencePanel from "@/components/EvidencePanel";
import StepTracker from "@/components/StepTracker";
import Findings from "@/components/Findings";

export default function Page() {
  const [queries, setQueries] = useState<CatalogQuery[]>([]);
  const [queryId, setQueryId] = useState<string | null>(null);
  const [typed, setTyped] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const run = useRun(1200);

  useEffect(() => {
    fetchCatalog()
      .then((c) => {
        setQueries(c.queries);
        if (c.queries[0]) setQueryId(c.queries[0].id);
      })
      .catch(() => setErr("Can't reach the API. Start it: cd api && uvicorn server:app --port 8000"));
  }, []);

  // Auto-run the canned query on first load and whenever it changes (the public URL self-plays —
  // D-DR7). Free text never auto-runs; it submits explicitly via the Ask button.
  useEffect(() => {
    if (!queryId) return;
    run.run({ kind: "canned", queryId });
    return () => run.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queryId]);

  const selected = queries.find((q) => q.id === queryId);
  const clipId = selected?.clip ?? "single-goal";
  const program = run.meta?.program ?? [];
  const currentStep = run.current >= 0 ? run.steps[run.current] ?? null : null;
  const total = run.meta?.total_steps ?? program.length;

  const askFreeText = () => {
    const text = typed.trim();
    if (!text) return; // empty/whitespace submit is a no-op
    run.run({ kind: "free", text, clip: clipId });
  };

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
            <h1 className="query">{run.meta?.query ?? selected?.text ?? "Loading…"}</h1>
            <div className="controls">
              {queries.length > 1 && (
                <select className="select" value={queryId ?? ""} onChange={(e) => setQueryId(e.target.value)}>
                  {queries.map((q) => <option key={q.id} value={q.id}>{q.text}</option>)}
                </select>
              )}
              <button
                className="btn btn-run"
                onClick={() => queryId && run.run({ kind: "canned", queryId })}
                disabled={run.status === "running"}
              >
                {run.status === "running" ? "Running…" : run.status === "done" ? "Replay" : "Run"}
              </button>
            </div>
          </div>

          <form
            className="ask-row"
            onSubmit={(e) => { e.preventDefault(); askFreeText(); }}
          >
            <input
              className="ask-input"
              type="text"
              value={typed}
              onChange={(e) => setTyped(e.target.value)}
              maxLength={500}
              placeholder={selected?.text ? `Ask anything about this clip — e.g. "${selected.text}"` : "Ask anything about this clip…"}
              aria-label="Ask a free-text question about the clip"
            />
            <span className="ask-clip" title="The clip your question runs against">{clipId}</span>
            <button className="btn btn-ask" type="submit" disabled={run.status === "running" || !typed.trim()}>
              Ask
            </button>
          </form>
        </header>

        {err && <p style={{ color: "var(--amber)", marginTop: 16, fontFamily: "var(--ff-mono)", fontSize: 13 }}>{err}</p>}

        <div className="panels">
          <ProgramPanel program={program} steps={run.steps} current={run.current} />
          <EvidencePanel step={currentStep} />
        </div>

        <section className="tracker" aria-label="Execution pipeline">
          <div className="tracker-top">
            <Findings status={run.status} findings={run.findings} stepCount={run.steps.length} total={total} programSource={run.meta?.program_source} />
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
