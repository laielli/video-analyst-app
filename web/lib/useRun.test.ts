import { describe, it, expect, vi, afterEach } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { useRun, API_BASE, fetchCatalog } from "@/lib/useRun";
import { MockEventSource } from "@/vitest.setup";

// useRun connects to GET /api/run over SSE via the global EventSource (mocked in vitest.setup.ts)
// and accumulates the run-doc: meta -> step* -> findings -> done. We drive the lifecycle by
// emitting named events on the most-recently-constructed MockEventSource.

const es = () => MockEventSource.last;

describe("useRun", () => {
  it("run() with the default canned query opens an EventSource with the right URL and goes running", () => {
    const { result } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    expect(es().url).toBe(`${API_BASE}/api/run?pace_ms=0&query=hero-10-first-goal`);
    expect(result.current.status).toBe("running");
  });

  it("captures meta and auto-advances current as steps arrive", () => {
    const { result } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    act(() => es().emit("meta", { query: "q", total_steps: 2, program: [] }));
    expect(result.current.meta).toMatchObject({ query: "q", total_steps: 2 });

    act(() => es().emit("step", { id: "a", op: "detect", status: "done", source: "cached" }));
    act(() => es().emit("step", { id: "b", op: "count", status: "done", source: "live" }));
    expect(result.current.steps).toHaveLength(2);
    // auto-advance: current tracks the latest step index.
    expect(result.current.current).toBe(1);
  });

  it("findings then done sets status done, stores findings, and closes the source", () => {
    const { result } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    const source = es();
    act(() => source.emit("findings", { answer: "Yes", verdict: "Yes — #10 scored", grounded: true }));
    act(() => source.emit("done", { ok: true }));
    expect(result.current.status).toBe("done");
    expect(result.current.findings).toMatchObject({ answer: "Yes", grounded: true });
    expect(source.closed).toBe(true);
  });

  it("select() pins current after done and stops auto-advance", () => {
    const { result } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    const source = es();
    act(() => source.emit("step", { id: "a", op: "detect", status: "done", source: "cached" }));
    act(() => source.emit("step", { id: "b", op: "count", status: "done", source: "live" }));
    act(() => source.emit("findings", { grounded: true, verdict: "v" }));
    act(() => source.emit("done", { ok: true }));

    act(() => result.current.select(0));
    expect(result.current.current).toBe(0);
    // a further step must NOT advance current once the user has pinned.
    act(() => source.emit("step", { id: "c", op: "answer", status: "done", source: "live" }));
    expect(result.current.current).toBe(0);
  });

  it("server-sent error sets status error with the message", () => {
    const { result } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    act(() => es().emitError({ message: "stream failed" }));
    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe("stream failed");
  });

  it("transport error (no data) sets 'connection lost'", () => {
    const { result } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    act(() => es().emitError(null));
    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe("connection lost");
  });

  it("error after done is ignored (done wins)", () => {
    const { result } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    const source = es();
    act(() => source.emit("findings", { grounded: true, verdict: "v" }));
    act(() => source.emit("done", { ok: true }));
    act(() => source.emitError(null));
    expect(result.current.status).toBe("done");
    expect(result.current.error).toBeNull();
  });

  it("free-text request URL carries query_text and clip", () => {
    const { result } = renderHook(() => useRun(null, 0));
    act(() => result.current.run({ kind: "free", text: "how many players?", clip: "single-goal" }));
    expect(es().url).toBe(
      `${API_BASE}/api/run?pace_ms=0&query_text=how%20many%20players%3F&clip=single-goal`,
    );
  });

  it("empty/whitespace free text is a no-op (no EventSource constructed)", () => {
    const { result } = renderHook(() => useRun(null, 0));
    act(() => result.current.run({ kind: "free", text: "   " }));
    expect(MockEventSource.instances).toHaveLength(0);
    expect(result.current.status).toBe("idle");
  });

  it("permalink request URL carries the run param (pace_ms first, then run)", () => {
    const { result } = renderHook(() => useRun(null, 0));
    act(() => result.current.run({ kind: "permalink", runId: "abc" }));
    // match runUrl's actual ordering: pace_ms first, then the request param.
    expect(es().url).toBe(`${API_BASE}/api/run?pace_ms=0&run=abc`);
    expect(result.current.status).toBe("running");
  });

  it("empty permalink runId is a no-op (no EventSource constructed)", () => {
    const { result } = renderHook(() => useRun(null, 0));
    act(() => result.current.run({ kind: "permalink", runId: "  " }));
    expect(MockEventSource.instances).toHaveLength(0);
    expect(result.current.status).toBe("idle");
  });

  it("meta run_id and cached are captured into state", () => {
    const { result } = renderHook(() => useRun(null, 0));
    act(() => result.current.run({ kind: "permalink", runId: "abc" }));
    act(() => es().emit("meta", { query: "q", total_steps: 0, program: [], run_id: "abc", cached: true }));
    expect(result.current.meta?.run_id).toBe("abc");
    expect(result.current.meta?.cached).toBe(true);
  });

  it("bare run() with no default query is a no-op", () => {
    const { result } = renderHook(() => useRun(null, 0));
    act(() => result.current.run());
    expect(MockEventSource.instances).toHaveLength(0);
  });

  it("unmount closes the source (cleanup effect)", () => {
    const { result, unmount } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    const source = es();
    expect(source.closed).toBe(false);
    unmount();
    expect(source.closed).toBe(true);
  });
});

// The multi-clip / multi-query catalog: fetchCatalog surfaces >1 query, so page.tsx renders the
// <select> (queries.length > 1) and the second clip is reachable through the product surface. We
// stub the global fetch (jsdom has no real network) to return a multi-query catalog.
describe("fetchCatalog (multi-query catalog)", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("surfaces more than one query and the second clip", async () => {
    const catalog = {
      clips: [
        { id: "single-goal", label: "Messi vs Mexico — World Cup 2022" },
        { id: "bernabeu-counter", label: "Real Madrid counter — Champions League 2025" },
      ],
      queries: [
        { id: "hero-10-first-goal", text: "Does #10 score the first goal?", clip: "single-goal" },
        { id: "bernabeu-count-players", text: "How many players are visible?", clip: "bernabeu-counter" },
        { id: "bernabeu-7-first-goal", text: "Does #7 score the first goal?", clip: "bernabeu-counter" },
      ],
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        expect(url).toBe(`${API_BASE}/api/catalog`);
        return { ok: true, json: async () => catalog } as Response;
      }),
    );

    const { queries } = await fetchCatalog();
    expect(queries.length).toBeGreaterThan(1);
    // the second clip is reachable: at least one query binds to it.
    expect(queries.some((q) => q.clip === "bernabeu-counter")).toBe(true);
  });

  it("throws on a non-ok catalog response", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 503 } as Response)));
    await expect(fetchCatalog()).rejects.toThrow("catalog 503");
  });
});
