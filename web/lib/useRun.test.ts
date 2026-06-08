import { describe, it, expect, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useRun, API_BASE } from "./useRun";
import { MockEventSource } from "../vitest.setup";

// The MockEventSource double (installed via vitest.setup.ts) records the constructed url,
// captures the named listeners useRun registers, and exposes emit()/emitError() to drive them.

function es(): MockEventSource {
  const s = MockEventSource.last();
  if (!s) throw new Error("no EventSource constructed");
  return s;
}

const HERO = "hero-10-first-goal";

beforeEach(() => {
  MockEventSource.instances = [];
});

describe("useRun — request URL + status", () => {
  it("bare run() replays the default canned query with the right URL and goes running", () => {
    const { result } = renderHook(() => useRun(HERO, 0));
    act(() => result.current.run());
    expect(MockEventSource.instances).toHaveLength(1);
    expect(es().url).toBe(`${API_BASE}/api/run?pace_ms=0&query=${HERO}`);
    expect(result.current.status).toBe("running");
  });

  it("free-text request carries query_text + clip", () => {
    const { result } = renderHook(() => useRun(null, 0));
    act(() => result.current.run({ kind: "free", text: "who scored?", clip: "single-goal" }));
    const url = es().url;
    expect(url).toContain("query_text=who%20scored%3F");
    expect(url).toContain("clip=single-goal");
  });

  it("empty/whitespace free text is a no-op — no EventSource constructed", () => {
    const { result } = renderHook(() => useRun(null, 0));
    act(() => result.current.run({ kind: "free", text: "   " }));
    expect(MockEventSource.instances).toHaveLength(0);
  });

  it("bare run() with no default query is a no-op", () => {
    const { result } = renderHook(() => useRun(null, 0));
    act(() => result.current.run());
    expect(MockEventSource.instances).toHaveLength(0);
  });
});

describe("useRun — event handling", () => {
  it("meta sets state.meta", () => {
    const { result } = renderHook(() => useRun(HERO, 0));
    act(() => result.current.run());
    const meta = { query: "q", clip: {}, program: [], total_steps: 2, pace_ms: 0 };
    act(() => es().emit("meta", meta));
    expect(result.current.meta).toMatchObject({ total_steps: 2, query: "q" });
  });

  it("two steps accumulate and auto-advance current to the latest", () => {
    const { result } = renderHook(() => useRun(HERO, 0));
    act(() => result.current.run());
    act(() => es().emit("step", { id: "a", op: "detect", status: "done", source: "cached" }));
    act(() => es().emit("step", { id: "b", op: "crop", status: "done", source: "live" }));
    expect(result.current.steps).toHaveLength(2);
    expect(result.current.current).toBe(1);
  });

  it("findings then done sets status=done, findings, and closes the source", () => {
    const { result } = renderHook(() => useRun(HERO, 0));
    act(() => result.current.run());
    const src = es();
    act(() => src.emit("findings", { answer: "Yes", verdict: "Yes — #10 scored", grounded: true }));
    act(() => src.emit("done", { ok: true }));
    expect(result.current.status).toBe("done");
    expect(result.current.findings).toMatchObject({ answer: "Yes", grounded: true });
    expect(src.closed).toBe(true);
  });

  it("select() pins current and stops auto-advance on later steps", () => {
    const { result } = renderHook(() => useRun(HERO, 0));
    act(() => result.current.run());
    const src = es();
    act(() => src.emit("step", { id: "a", op: "detect", status: "done", source: "cached" }));
    act(() => src.emit("findings", { grounded: true, answer: "Yes", verdict: "v" }));
    act(() => src.emit("done", { ok: true }));
    act(() => result.current.select(0));
    expect(result.current.current).toBe(0);
    // a further step must NOT advance current (user has pinned).
    act(() => src.emit("step", { id: "b", op: "crop", status: "done", source: "live" }));
    expect(result.current.current).toBe(0);
  });
});

describe("useRun — error handling", () => {
  it("server-sent error event (has data) -> status=error with the message", () => {
    const { result } = renderHook(() => useRun(HERO, 0));
    act(() => result.current.run());
    act(() => es().emitError({ message: "stream failed" }));
    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe("stream failed");
  });

  it("transport error (no data) -> 'connection lost'", () => {
    const { result } = renderHook(() => useRun(HERO, 0));
    act(() => result.current.run());
    act(() => es().emitError(null));
    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe("connection lost");
  });

  it("error after done is ignored (stays done)", () => {
    const { result } = renderHook(() => useRun(HERO, 0));
    act(() => result.current.run());
    const src = es();
    act(() => src.emit("findings", { grounded: true, answer: "Yes", verdict: "v" }));
    act(() => src.emit("done", { ok: true }));
    act(() => src.emitError(null));
    expect(result.current.status).toBe("done");
  });
});

describe("useRun — lifecycle", () => {
  it("unmount closes the source", () => {
    const { result, unmount } = renderHook(() => useRun(HERO, 0));
    act(() => result.current.run());
    const src = es();
    expect(src.closed).toBe(false);
    unmount();
    expect(src.closed).toBe(true);
  });

  it("a new run() closes the prior source before opening a new one", () => {
    const { result } = renderHook(() => useRun(HERO, 0));
    act(() => result.current.run());
    const first = es();
    act(() => result.current.run());
    expect(first.closed).toBe(true);
    expect(MockEventSource.instances).toHaveLength(2);
  });
});
