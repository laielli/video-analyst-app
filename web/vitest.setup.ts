import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

// ---- EventSource test double -----------------------------------------------------------------
// jsdom does NOT implement EventSource, which useRun.ts constructs. This controllable double
// records the url, captures addEventListener handlers, and exposes emit()/emitError() so tests
// drive the SSE stream synchronously. The hook only touches `new EventSource(url)`,
// `addEventListener(name, cb)`, and `close()` — it never reads readyState, uses onmessage, or
// removes listeners — so the double stays minimal.

export class MockEventSource {
  static instances: MockEventSource[] = [];

  url: string;
  closed = false;
  listeners: Record<string, ((e: any) => void)[]> = {};

  constructor(url: string) {
    this.url = url;
    MockEventSource.instances.push(this);
  }

  addEventListener(name: string, cb: (e: any) => void) {
    (this.listeners[name] ??= []).push(cb);
  }

  close() {
    this.closed = true;
  }

  /** Dispatch a named server event carrying a JSON-encoded data payload. */
  emit(name: string, data: unknown) {
    const e = { data: JSON.stringify(data) };
    (this.listeners[name] ?? []).forEach((cb) => cb(e));
  }

  /** Dispatch an `error` event. Pass null/undefined for a transport error (no data). */
  emitError(data: unknown = null) {
    const e = { data: data == null ? undefined : JSON.stringify(data) };
    (this.listeners["error"] ?? []).forEach((cb) => cb(e));
  }

  /** The most recently constructed instance (the one the latest run() created). */
  static last(): MockEventSource {
    return MockEventSource.instances[MockEventSource.instances.length - 1];
  }

  static reset() {
    MockEventSource.instances = [];
  }
}

vi.stubGlobal("EventSource", MockEventSource as unknown as typeof EventSource);

afterEach(() => {
  cleanup();
  MockEventSource.reset();
});
