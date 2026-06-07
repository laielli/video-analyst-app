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
  const run = useRun(queryId, 1200);

  useEffect(() => {
    fetchCatalog()
      .then((c) => {
        setQueries(c.queries);
        if (c.queries[0]) setQueryId(c.queries[0].id);
      })
      .catch(() => setErr("Can't reach the API. Start it: cd api && uvicorn server:app --port 8000"));
  }, []);

  // Auto-run on first load and whenever the canned query changes (the public URL self-plays — D-DR7).
  useEffect(() => {
    if (!queryId) return;
    run.run();
    return () => run.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queryId]);

  const selected = queries.find((q) => q.id === queryId);
  // v1 has one clip; the free-text run targets the selected canned query's clip.
  const clipId = selected?.clip ?? "single-goal";

  // Submit a typed question -> live codegen + execution (Resolved Q1 -> A on a miss).
  const ask = () => {
    const text = typed.trim();
    if (!text) return; // empty/whitespace is a no-op
    run.run({ kind: "free", text, clip: clipId });
  };

  // The active question: the running run-doc's query (free text or canned) else the selected text.
  const activeQuery = run.meta?.query ?? selected?.text ?? "Loading…";
  const program = run.meta?.program ?? [];
  const currentStep = run.current >= 0 ? run.steps[run.current] ?? null : null;
  const total = run.meta?.total_steps ?? program.length;

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
            <h1 className="query">{activeQuery}</h1>
            <div className="controls">
              {queries.length > 1 && (
                <select className="select" value={queryId ?? ""} onChange={(e) => setQueryId(e.target.value)}>
                  {queries.map((q) => <option key={q.id} value={q.id}>{q.text}</option>)}
                </select>
              )}
              <button className="btn btn-run" onClick={() => run.run()} disabled={run.status === "running"}>
                {run.status === "running" ? "Running…" : run.status === "done" ? "Replay" : "Run"}
              </button>
            </div>
          </div>
          <form
            className="ask-row"
            onSubmit={(e) => { e.preventDefault(); ask(); }}
          >
            <input
              className="ask-input"
              type="text"
              value={typed}
              onChange={(e) => setTyped(e.target.value)}
              placeholder={selected?.text ? `Ask anything, e.g. "${selected.text}"` : "Ask a question about the clip…"}
              maxLength={500}
              aria-label="Ask a question about the clip"
              disabled={run.status === "running"}
            />
            <span className="ask-clip" title="Clip context (v1 has one clip)">{clipId}</span>
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
            <div className="findings-row">
              <Findings status={run.status} findings={run.findings} stepCount={run.steps.length} total={total} />
              {run.meta?.program_source && (
                <span
                  className={`prov-tag ${run.meta.program_source}`}
                  title="Provenance of the program: live = compiled this run by Azure OpenAI codegen; pinned = known-good fallback; ungrounded = couldn't ground a free-text query (no hero substitution)."
                >
                  {run.meta.program_source === "live"
                    ? "live compile"
                    : run.meta.program_source === "ungrounded"
                    ? "ungrounded"
                    : "pinned"}
                </span>
              )}
            </div>
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
