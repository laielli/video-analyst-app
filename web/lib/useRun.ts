"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Meta, StepResult, Findings, CatalogQuery } from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export type RunStatus = "idle" | "running" | "done" | "error";

export type RunState = {
  status: RunStatus;
  meta: Meta | null;
  steps: StepResult[]; // trace entries as they arrive
  findings: Findings | null;
  error: string | null;
  /** index into steps the UI is showing (latest while running; user-selectable after). */
  current: number;
};

/** A run request: either a canned query id, or a typed free-text question for a clip. */
export type RunRequest =
  | { kind: "canned"; queryId: string }
  | { kind: "free"; text: string; clip: string };

/** Whether a findings payload is the honest "couldn't ground this" state (Q1 -> A). */
export function isUngrounded(findings: Findings | null): boolean {
  return findings?.grounded === false;
}

function runUrl(req: RunRequest, paceMs: number): string {
  const q =
    req.kind === "free"
      ? `query_text=${encodeURIComponent(req.text)}&clip=${encodeURIComponent(req.clip)}`
      : `query=${encodeURIComponent(req.queryId)}`;
  return `${API_BASE}/api/run?${q}&pace_ms=${paceMs}`;
}

/**
 * Connects to GET /api/run over SSE and accumulates the run-doc step by step.
 * meta -> step* -> findings -> done. After done, `select()` scrubs to any step.
 *
 * `run(req)` takes a RunRequest (canned id or free-text). The hook holds no query identity of
 * its own; the caller supplies the request at run time so canned + free text share one path.
 */
export function useRun(paceMs = 1200) {
  const [state, setState] = useState<RunState>({
    status: "idle", meta: null, steps: [], findings: null, error: null, current: -1,
  });
  const esRef = useRef<EventSource | null>(null);
  const userPinnedRef = useRef(false); // true once the user scrubs, so we stop auto-advancing

  const stop = useCallback(() => {
    esRef.current?.close();
    esRef.current = null;
  }, []);

  const run = useCallback((req: RunRequest) => {
    if (req.kind === "canned" && !req.queryId) return;
    if (req.kind === "free" && !req.text.trim()) return;
    stop();
    userPinnedRef.current = false;
    setState({ status: "running", meta: null, steps: [], findings: null, error: null, current: -1 });

    const es = new EventSource(runUrl(req, paceMs));
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
  }, [paceMs, stop]);

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
