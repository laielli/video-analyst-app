"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Meta, StepResult, Findings, CatalogQuery } from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export type RunStatus = "idle" | "running" | "done" | "error";

/**
 * What to run: a canned query by id, or a typed free-text question against a clip (D-DR4).
 * Free text builds `?query_text=&clip=`; canned builds `?query=`.
 */
export type RunRequest =
  | { kind: "canned"; queryId: string }
  | { kind: "free"; text: string; clip: string };

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
  const base = `${API_BASE}/api/run`;
  if (req.kind === "free") {
    return `${base}?query_text=${encodeURIComponent(req.text)}&clip=${encodeURIComponent(req.clip)}&pace_ms=${paceMs}`;
  }
  return `${base}?query=${encodeURIComponent(req.queryId)}&pace_ms=${paceMs}`;
}

/**
 * Connects to GET /api/run over SSE and accumulates the run-doc step by step.
 * meta -> step* -> findings -> done. After done, `select()` scrubs to any step.
 *
 * `run()` with no argument replays the canned `queryId` the hook was constructed with (keeps the
 * D-DR7 auto-play URL working); `run(request)` runs an explicit canned/free-text request.
 */
export function useRun(queryId: string | null, paceMs = 1200) {
  const [state, setState] = useState<RunState>({
    status: "idle", meta: null, steps: [], findings: null, error: null, current: -1,
  });
  const esRef = useRef<EventSource | null>(null);
  const userPinnedRef = useRef(false); // true once the user scrubs, so we stop auto-advancing

  const stop = useCallback(() => {
    esRef.current?.close();
    esRef.current = null;
  }, []);

  const run = useCallback((request?: RunRequest) => {
    const req: RunRequest | null =
      request ?? (queryId ? { kind: "canned", queryId } : null);
    if (!req) return;
    stop();
    userPinnedRef.current = false;
    setState({ status: "running", meta: null, steps: [], findings: null, error: null, current: -1 });

    const url = runUrl(req, paceMs);
    const es = new EventSource(url);
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
  }, [queryId, paceMs, stop]);

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
