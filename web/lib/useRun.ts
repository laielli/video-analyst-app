"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Meta, StepResult, Findings, CatalogQuery } from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export type RunStatus = "idle" | "running" | "done" | "error";

/**
 * A run request: either a canned query (by id) or a free-text query (typed `text` against a
 * `clip`). `useRun.run` accepts either; a bare `run()` with no arg replays the hook's default
 * canned query (the auto-play URL — D-DR7).
 */
export type RunRequest =
  | { kind: "canned"; queryId: string }
  | { kind: "free"; text: string; clip?: string }
  // A shareable permalink replay: stream a STORED run by its run_id (?run=<id>). The id is the
  // backend content-address, so a permalink is stable across processes (replays even cold).
  | { kind: "permalink"; runId: string };

export type RunState = {
  status: RunStatus;
  meta: Meta | null;
  steps: StepResult[]; // trace entries as they arrive
  findings: Findings | null;
  error: string | null;
  /** index into steps the UI is showing (latest while running; user-selectable after). */
  current: number;
};

function runUrl(req: RunRequest, paceMs: number): string {
  const base = `${API_BASE}/api/run?pace_ms=${paceMs}`;
  if (req.kind === "canned") return `${base}&query=${encodeURIComponent(req.queryId)}`;
  if (req.kind === "permalink") return `${base}&run=${encodeURIComponent(req.runId)}`;
  const clip = req.clip ? `&clip=${encodeURIComponent(req.clip)}` : "";
  return `${base}&query_text=${encodeURIComponent(req.text)}${clip}`;
}

/**
 * Connects to GET /api/run over SSE and accumulates the run-doc step by step.
 * meta -> step* -> findings -> done. After done, `select()` scrubs to any step.
 *
 * `defaultQueryId` is the canned query a bare `run()` (auto-play) replays. `run(req)` runs an
 * explicit request — canned or free-text — so one hook serves both the self-playing public URL
 * and the typed-question path.
 */
export function useRun(defaultQueryId: string | null, paceMs = 1200) {
  const [state, setState] = useState<RunState>({
    status: "idle", meta: null, steps: [], findings: null, error: null, current: -1,
  });
  const esRef = useRef<EventSource | null>(null);
  const userPinnedRef = useRef(false); // true once the user scrubs, so we stop auto-advancing

  const stop = useCallback(() => {
    esRef.current?.close();
    esRef.current = null;
  }, []);

  const run = useCallback((req?: RunRequest) => {
    // No explicit request -> replay the default canned query (the auto-play URL).
    const request: RunRequest | null =
      req ?? (defaultQueryId ? { kind: "canned", queryId: defaultQueryId } : null);
    if (!request) return;
    if (request.kind === "free" && !request.text.trim()) return; // empty/whitespace -> no-op
    if (request.kind === "permalink" && !request.runId.trim()) return; // no id -> no-op
    stop();
    userPinnedRef.current = false;
    setState({ status: "running", meta: null, steps: [], findings: null, error: null, current: -1 });

    const es = new EventSource(runUrl(request, paceMs));
    esRef.current = es;

    es.addEventListener("meta", (e) => {
      setState((s) => ({ ...s, meta: JSON.parse((e as MessageEvent).data) }));
    });
    es.addEventListener("step", (e) => {
      const step = JSON.parse((e as MessageEvent).data);
      setState((s) => {
        const steps = [...s.steps, step];
        return { ...s, steps, current: userPinnedRef.current ? s.current : steps.length - 1 };
      });
    });
    es.addEventListener("findings", (e) => {
      setState((s) => ({ ...s, findings: JSON.parse((e as MessageEvent).data) }));
    });
    es.addEventListener("done", () => {
      setState((s) => ({ ...s, status: "done" }));
      stop();
    });
    es.addEventListener("error", (e) => {
      // Distinguish a server-sent `error` event (has data) from a transport error.
      const data = (e as MessageEvent).data;
      setState((s) =>
        s.status === "done"
          ? s
          : { ...s, status: "error", error: data ? JSON.parse(data).message : "connection lost" }
      );
      stop();
    });
  }, [defaultQueryId, paceMs, stop]);

  /** Scrub to a step after the run completes (D-DR3 click-to-replay). */
  const select = useCallback((i: number) => {
    userPinnedRef.current = true;
    setState((s) => ({ ...s, current: Math.max(0, Math.min(i, s.steps.length - 1)) }));
  }, []);

  useEffect(() => () => stop(), [stop]);

  return { ...state, run, stop, select };
}

export async function fetchCatalog(): Promise<{ queries: CatalogQuery[] }> {
  const res = await fetch(`${API_BASE}/api/catalog`);
  if (!res.ok) throw new Error(`catalog ${res.status}`);
  return res.json();
}
