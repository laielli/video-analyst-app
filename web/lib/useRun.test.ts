import { describe, it, expect } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useRun, API_BASE } from "./useRun";
import { MockEventSource } from "../vitest.setup";

// The MockEventSource is installed globally in vitest.setup.ts and reset after each test.
const es = () => MockEventSource.last();

function metaPayload(overrides: Record<string, unknown> = {}) {
  return {
    query: "q", clip: { id: "c", width: 1, height: 1, duration_ms: 1 },
    program: [], total_steps: 2, pace_ms: 0, program_source: "pinned", ...overrides,
  };
}
const stepPayload = (id: string) => ({ id, op: "detect", status: "done", source: "cached" });

describe("useRun", () => {
  it("run() with no arg replays the default canned query and goes running", () => {
    const { result } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    expect(es().url).toBe(`${API_BASE}/api/run?pace_ms=0&query=hero-10-first-goal`);
    expect(result.current.status).toBe("running");
    expect(MockEventSource.instances).toHaveLength(1);
  });

  it("meta sets state.meta; two steps advance current and accumulate", () => {
    const { result } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    act(() => es().emit("meta", metaPayload()));
    expect(result.current.meta?.total_steps).toBe(2);
    act(() => es().emit("step", stepPayload("s1")));
    act(() => es().emit("step", stepPayload("s2")));
    expect(result.current.steps).toHaveLength(2);
    expect(result.current.current).toBe(1); // auto-advance to latest
  });

  it("findings then done -> status done, findings set, source closed", () => {
    const { result } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    const source = es();
    act(() => source.emit("findings", { grounded: true, verdict: "Yes" }));
    act(() => source.emit("done", { ok: true }));
    expect(result.current.status).toBe("done");
    expect(result.current.findings?.verdict).toBe("Yes");
    expect(source.closed).toBe(true);
  });

  it("select(i) pins current and stops auto-advance for later steps", () => {
    const { result } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    act(() => es().emit("step", stepPayload("s1")));
    act(() => es().emit("step", stepPayload("s2")));
    act(() => es().emit("done", { ok: true }));
    act(() => result.current.select(0));
    expect(result.current.current).toBe(0);
    // a further step must NOT move current off the pinned index.
    act(() => es().emit("step", stepPayload("s3")));
    expect(result.current.current).toBe(0);
  });

  it("server error event with data sets status error + message", () => {
    const { result } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    act(() => es().emit("meta", metaPayload()));
    act(() => es().emitError({ message: "stream failed" }));
    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe("stream failed");
  });

  it("transport error (no data) -> 'connection lost'", () => {
    const { result } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    act(() => es().emitError(null));
    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe("connection lost");
  });

  it("error after done is ignored (stays done)", () => {
    const { result } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    const source = es();
    act(() => source.emit("done", { ok: true }));
    act(() => source.emitError(null));
    expect(result.current.status).toBe("done");
    expect(result.current.error).toBeNull();
  });

  it("free-text request builds a query_text + clip URL", () => {
    const { result } = renderHook(() => useRun(null, 0));
    act(() => result.current.run({ kind: "free", text: "how many?", clip: "single-goal" }));
    expect(es().url).toBe(
      `${API_BASE}/api/run?pace_ms=0&query_text=how%20many%3F&clip=single-goal`
    );
  });

  it("empty/whitespace free-text is a no-op (no EventSource)", () => {
    const { result } = renderHook(() => useRun(null, 0));
    act(() => result.current.run({ kind: "free", text: "   " }));
    expect(MockEventSource.instances).toHaveLength(0);
    expect(result.current.status).toBe("idle");
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
