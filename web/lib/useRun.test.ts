import { describe, it, expect } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { useRun, API_BASE } from "@/lib/useRun";
import { MockEventSource } from "@/vitest.setup";

// useRun connects to GET /api/run over SSE and accumulates the run-doc. The mock EventSource
// (installed in vitest.setup.ts) lets us drive named events. The hook only touches
// `new EventSource(url)`, `addEventListener(name, cb)`, and `close()`.

const META = { query: "q", clip: {}, program: [], total_steps: 2, pace_ms: 0, program_source: "pinned" };
const STEP = (id: string) => ({ id, op: "detect", status: "done", source: "live" });

describe("useRun", () => {
  it("run() with the default canned query builds the right URL and goes running", () => {
    const { result } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    const es = MockEventSource.last()!;
    expect(es.url).toBe(`${API_BASE}/api/run?pace_ms=0&query=hero-10-first-goal`);
    expect(result.current.status).toBe("running");
  });

  it("meta sets state.meta; two steps auto-advance current to the latest", () => {
    const { result } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    const es = MockEventSource.last()!;
    act(() => es.emit("meta", META));
    expect(result.current.meta).toEqual(META);
    act(() => es.emit("step", STEP("s0")));
    act(() => es.emit("step", STEP("s1")));
    expect(result.current.steps).toHaveLength(2);
    expect(result.current.current).toBe(1); // auto-advanced
  });

  it("findings then done -> status done, findings set, source closed", () => {
    const { result } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    const es = MockEventSource.last()!;
    const findings = { answer: "Yes", verdict: "Yes — #10 scored", grounded: true };
    act(() => es.emit("findings", findings));
    act(() => es.emit("done", { ok: true }));
    expect(result.current.status).toBe("done");
    expect(result.current.findings).toEqual(findings);
    expect(es.closed).toBe(true);
  });

  it("select() pins current and stops auto-advance", () => {
    const { result } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    const es = MockEventSource.last()!;
    act(() => es.emit("step", STEP("s0")));
    act(() => es.emit("step", STEP("s1")));
    act(() => es.emit("findings", { grounded: true }));
    act(() => es.emit("done", { ok: true }));
    // pin to step 0.
    act(() => result.current.select(0));
    expect(result.current.current).toBe(0);
    // a further step must NOT advance current (still pinned at 0).
    act(() => es.emit("step", STEP("s2")));
    expect(result.current.current).toBe(0);
  });

  it("server-sent error -> status error with message", () => {
    const { result } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    const es = MockEventSource.last()!;
    act(() => es.emitError({ message: "stream failed" }));
    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe("stream failed");
  });

  it("transport error (no data) -> 'connection lost'", () => {
    const { result } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    const es = MockEventSource.last()!;
    act(() => es.emitError(null));
    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe("connection lost");
  });

  it("error after done is ignored", () => {
    const { result } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    const es = MockEventSource.last()!;
    act(() => es.emit("done", { ok: true }));
    act(() => es.emitError({ message: "late error" }));
    expect(result.current.status).toBe("done"); // stays done
    expect(result.current.error).toBeNull();
  });

  it("free-text request URL carries query_text and clip", () => {
    const { result } = renderHook(() => useRun(null, 0));
    act(() => result.current.run({ kind: "free", text: "how many?", clip: "single-goal" }));
    const es = MockEventSource.last()!;
    expect(es.url).toBe(
      `${API_BASE}/api/run?pace_ms=0&query_text=how%20many%3F&clip=single-goal`
    );
  });

  it("empty/whitespace free text is a no-op (no EventSource)", () => {
    const { result } = renderHook(() => useRun(null, 0));
    act(() => result.current.run({ kind: "free", text: "   " }));
    expect(MockEventSource.instances).toHaveLength(0);
    expect(result.current.status).toBe("idle");
  });

  it("unmount closes the source", () => {
    const { result, unmount } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    const es = MockEventSource.last()!;
    unmount();
    expect(es.closed).toBe(true);
  });
});
