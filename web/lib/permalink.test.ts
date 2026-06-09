import { describe, it, expect } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { useRun, API_BASE, runLink } from "@/lib/useRun";
import { MockEventSource } from "@/vitest.setup";

// Permalink / shareable-replay coverage at the hook + pure-helper level. Page-level URL-param
// load behavior (?run=, ?query=) lives in app/page.test.tsx.

const es = () => MockEventSource.last;

describe("permalink request", () => {
  it("permalink request URL carries the run param", () => {
    const { result } = renderHook(() => useRun(null, 0));
    act(() => result.current.run({ kind: "permalink", runId: "abc" }));
    // match runUrl's actual ordering: pace_ms first, then the request param.
    expect(es().url).toBe(`${API_BASE}/api/run?pace_ms=0&run=abc`);
    expect(result.current.status).toBe("running");
  });

  it("meta run_id is captured into state", () => {
    const { result } = renderHook(() => useRun(null, 0));
    act(() => result.current.run({ kind: "permalink", runId: "abc" }));
    const runId = "a".repeat(64);
    act(() =>
      es().emit("meta", { query: "q", total_steps: 0, program: [], run_id: runId, cached: true }),
    );
    expect(result.current.meta?.run_id).toBe(runId);
    expect(result.current.meta?.cached).toBe(true);
  });
});

describe("runLink helper", () => {
  it("copy-link is built from origin + run_id", () => {
    const runId = "a".repeat(64);
    expect(runLink("https://demo.example", runId)).toBe(`https://demo.example/?run=${runId}`);
  });

  it("encodes the run id (defense in depth, even though ids are hex)", () => {
    expect(runLink("https://x.test", "a b")).toBe("https://x.test/?run=a%20b");
  });
});
