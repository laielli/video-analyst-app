import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

// A controllable EventSource test double. jsdom does NOT implement EventSource, and useRun.ts
// only touches `new EventSource(url)`, `addEventListener(name, cb)`, and `close()` — it never
// reads readyState, never uses onmessage, never removes listeners. The double records the url,
// captures handlers per event name, and exposes emit()/emitError() to drive the hook.
export class MockEventSource {
  static instances: MockEventSource[] = [];

  url: string;
  closed = false;
  private listeners: Record<string, Array<(e: MessageEvent) => void>> = {};

  constructor(url: string) {
    this.url = url;
    MockEventSource.instances.push(this);
  }

  addEventListener(name: string, cb: (e: MessageEvent) => void) {
    (this.listeners[name] ??= []).push(cb);
  }

  close() {
    this.closed = true;
  }

  /** Dispatch a named server event carrying a JSON-serializable data payload. */
  emit(name: string, data: unknown) {
    const e = { data: JSON.stringify(data) } as MessageEvent;
    (this.listeners[name] ?? []).forEach((cb) => cb(e));
  }

  /** Dispatch an `error` event. Pass null for a transport error (no data field). */
  emitError(data: unknown | null = null) {
    const e = { data: data == null ? null : JSON.stringify(data) } as MessageEvent;
    (this.listeners["error"] ?? []).forEach((cb) => cb(e));
  }

  /** The most recently constructed instance (the active EventSource for a single run). */
  static last(): MockEventSource | undefined {
    return MockEventSource.instances[MockEventSource.instances.length - 1];
  }
}

vi.stubGlobal("EventSource", MockEventSource as unknown as typeof EventSource);

afterEach(() => {
  cleanup();
  MockEventSource.instances = [];
});
