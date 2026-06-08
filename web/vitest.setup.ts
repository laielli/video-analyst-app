import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

// jsdom does not implement EventSource, and useRun.ts constructs one directly. Install a
// controllable test double. The hook only touches: new EventSource(url), addEventListener(name,
// cb), and close() — it never reads readyState, never uses onmessage, never removes listeners.
// So the double records `url`, captures handlers per event name, tracks close(), and exposes
// emit(name, dataObj) / emitError(dataOrNull) so tests can drive the SSE lifecycle.

export class MockEventSource {
  static instances: MockEventSource[] = [];
  /** Most-recently constructed instance — the one the current run() call is wired to. */
  static get last(): MockEventSource {
    return MockEventSource.instances[MockEventSource.instances.length - 1];
  }
  static reset() {
    MockEventSource.instances = [];
  }

  url: string;
  closed = false;
  private handlers: Record<string, ((e: MessageEvent) => void)[]> = {};

  constructor(url: string) {
    this.url = url;
    MockEventSource.instances.push(this);
  }

  addEventListener(name: string, cb: (e: MessageEvent) => void) {
    (this.handlers[name] ??= []).push(cb);
  }

  close() {
    this.closed = true;
  }

  /** Fire a named server event carrying JSON `data`. */
  emit(name: string, data: unknown) {
    const evt = { data: JSON.stringify(data) } as MessageEvent;
    for (const cb of this.handlers[name] ?? []) cb(evt);
  }

  /** Fire an `error` event. Pass an object for a server-sent error (has data) or null for a
   * transport error (no data -> "connection lost"). */
  emitError(data: unknown | null) {
    const evt = { data: data == null ? undefined : JSON.stringify(data) } as MessageEvent;
    for (const cb of this.handlers["error"] ?? []) cb(evt);
  }
}

vi.stubGlobal("EventSource", MockEventSource as unknown as typeof EventSource);

afterEach(() => {
  cleanup();
  MockEventSource.reset();
});
