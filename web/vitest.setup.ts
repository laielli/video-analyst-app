import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

// jsdom does not implement EventSource, which useRun.ts constructs. Install a controllable double
// as the global so the SSE hook can be driven from tests. The hook only ever touches
// `new EventSource(url)`, `addEventListener(name, cb)`, and `close()` — nothing else — so the
// double stays deliberately small. `emit`/`emitError` let tests push named SSE events.

type Handler = (e: any) => void;

export class MockEventSource {
  // Every constructed instance is recorded so a test can grab the latest one.
  static instances: MockEventSource[] = [];

  url: string;
  closed = false;
  private handlers: Record<string, Handler[]> = {};

  constructor(url: string) {
    this.url = url;
    MockEventSource.instances.push(this);
  }

  addEventListener(name: string, cb: Handler) {
    (this.handlers[name] ??= []).push(cb);
  }

  close() {
    this.closed = true;
  }

  /** Push a named server-sent event carrying a JSON data payload. */
  emit(name: string, data: unknown) {
    const evt = { data: JSON.stringify(data) };
    (this.handlers[name] ?? []).forEach((cb) => cb(evt));
  }

  /** Fire the `error` listener. Pass an object for a server-sent error (has data), or null for a
   *  transport error (no data) — matching how useRun distinguishes the two. */
  emitError(data: unknown | null = null) {
    const evt = data == null ? { data: undefined } : { data: JSON.stringify(data) };
    (this.handlers["error"] ?? []).forEach((cb) => cb(evt));
  }

  static reset() {
    MockEventSource.instances = [];
  }

  /** The most recently constructed instance (the one the latest run() created). */
  static last(): MockEventSource | undefined {
    return MockEventSource.instances[MockEventSource.instances.length - 1];
  }
}

vi.stubGlobal("EventSource", MockEventSource);

afterEach(() => {
  cleanup();
  MockEventSource.reset();
});
