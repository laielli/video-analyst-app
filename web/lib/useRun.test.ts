import { describe, it, expect } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useRun, API_BASE } from "@/lib/useRun";
import { MockEventSource } from "@/vitest.setup";

// useRun connects to GET /api/run over SSE and accumulates the run-doc step by step. The
// EventSource is mocked (jsdom has none) — see vitest.setup.ts. We drive the stream with
// MockEventSource.last().emit(...) inside act() so React state updates flush.

const meta = (total = 2) => ({
  query: "q",
  clip: { id: "c", width: 1, height: 1, duration_ms: 1 },
  program: [],
  total_steps: total,
  pace_ms: 0,
  program_source: "pinned",
});
const step = (id: string) => ({ id, op: "detect", status: "done", source: "cached" });
const findings = { answer: "Yes", verdict: "Yes — ...", grounded: true };

describe("useRun", () => {
  it("run() with the default canned query builds the right URL and goes running", () => {
    const { result } = renderHook(() => useRun("hero-10-first-goal", 0));
    act(() => result.current.run());
    const es = MockEventSource.last();
    expect(es.url).toBe(`${API_BASE}/api/run?pace_ms=0&query=hero-10-first-goal`);
    expect(result.current.status).toBe("running");
  });

  it("emits meta then two steps -> steps.length 2, current auto-advances to 1", () => {
    const { result } = renderHook(() => useRun("hero", 0));
    act(() => result.current.run());
    const es = MockEventSource.last();
    act(() => es.emit("meta", meta(2)));
    expect(result.current.meta?.total_steps).toBe(2);
    act(() => es.emit("step", step("a")));
    act(() => es.emit("step", step("b")));
    expect(result.current.steps.length).toBe(2);
    expect(result.current.current).toBe(1); // auto-advance to latest
  });

  it("findings then done -> status done, findings set, source closed", () => {
    const { result } = renderHook(() => useRun("hero", 0));
    act(() => result.current.run());
    const es = MockEventSource.last();
    act(() => es.emit("findings", findings));
    act(() => es.emit("done", { ok: true }));
    expect(result.current.status).toBe("done");
    expect(result.current.findings?.verdict).toBe("Yes — ...");
    expect(es.closed).toBe(true);
  });

  it("select() after done pins current and stops auto-advance", () => {
    const { result } = renderHook(() => useRun("hero", 0));
    act(() => result.current.run());
    const es = MockEventSource.last();
    act(() => es.emit("step", step("a")));
    act(() => es.emit("step", step("b")));
    act(() => es.emit("done", { ok: true }));
    // pin to step 0
    act(() => result.current.select(0));
    expect(result.current.current).toBe(0);
    // a further step must NOT advance current (userPinned)
    act(() => es.emit("step", step("c")));
    expect(result.current.current).toBe(0);
    expect(result.current.steps.length).toBe(3);
  });

  it("server error event with data -> status error + message", () => {
    const { result } = renderHook(() => useRun("hero", 0));
    act(() => result.current.run());
    const es = MockEventSource.last();
    act(() => es.emitError({ message: "stream failed" }));
    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe("stream failed");
    expect(es.closed).toBe(true);
  });

  it("transport error (no data) -> 'connection lost'", () => {
    const { result } = renderHook(() => useRun("hero", 0));
    act(() => result.current.run());
    const es = MockEventSource.last();
    act(() => es.emitError(null));
    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe("connection lost");
  });

  it("error after done is ignored (done wins)", () => {
    const { result } = renderHook(() => useRun("hero", 0));
    act(() => result.current.run());
    const es = MockEventSource.last();
    act(() => es.emit("done", { ok: true }));
    act(() => es.emitError(null));
    expect(result.current.status).toBe("done");
    expect(result.current.error).toBeNull();
  });

  it("free-text request URL carries query_text + clip", () => {
    const { result } = renderHook(() => useRun(null, 0));
    act(() => result.current.run({ kind: "free", text: "how many players?", clip: "single-goal" }));
    const es = MockEventSource.last();
    expect(es.url).toBe(
      `${API_BASE}/api/run?pace_ms=0&query_text=how%20many%20players%3F&clip=single-goal`
    );
  });

  it("free-text without a clip omits the clip param", () => {
    const { result } = renderHook(() => useRun(null, 0));
    act(() => result.current.run({ kind: "free", text: "hi" }));
    expect(MockEventSource.last().url).toBe(`${API_BASE}/api/run?pace_ms=0&query_text=hi`);
  });

  it("empty/whitespace free text is a no-op (no EventSource constructed)", () => {
    MockEventSource.reset();
    const { result } = renderHook(() => useRun(null, 0));
    act(() => result.current.run({ kind: "free", text: "   " }));
    expect(MockEventSource.instances.length).toBe(0);
    expect(result.current.status).toBe("idle");
  });

  it("bare run() with no default query is a no-op", () => {
    MockEventSource.reset();
    const { result } = renderHook(() => useRun(null, 0));
    act(() => result.current.run());
    expect(MockEventSource.instances.length).toBe(0);
  });

  it("unmount closes the source (cleanup effect)", () => {
    const { result, unmount } = renderHook(() => useRun("hero", 0));
    act(() => result.current.run());
    const es = MockEventSource.last();
    unmount();
    expect(es.closed).toBe(true);
  });
});
