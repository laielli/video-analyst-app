import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

// A controllable EventSource test double. jsdom has no EventSource, and useRun.ts only ever
// touches: `new EventSource(url)`, `addEventListener(name, cb)`, and `close()`. It never reads
// readyState, never uses onmessage, never removes listeners — so the double stays minimal.
//
// Each constructed instance records its url, captures named listeners, and exposes:
//   emit(name, dataObj)  -> dispatch a server-sent named event carrying JSON `data`.
//   emitError(dataOrNull) -> dispatch an `error` event; pass null for a transport error (no data).
// MockEventSource.instances collects every instance so tests can grab the latest.
export class MockEventSource {
  static instances: MockEventSource[] = [];
  static last(): MockEventSource {
    return MockEventSource.instances[MockEventSource.instances.length - 1];
  }
  static reset() {
    MockEventSource.instances = [];
  }

  url: string;
  closed = false;
  listeners: Record<string, ((e: unknown) => void)[]> = {};

  constructor(url: string) {
    this.url = url;
    MockEventSource.instances.push(this);
  }

  addEventListener(name: string, cb: (e: unknown) => void) {
    (this.listeners[name] ??= []).push(cb);
  }

  close() {
    this.closed = true;
  }

  emit(name: string, dataObj: unknown) {
    const event = { data: JSON.stringify(dataObj) };
    (this.listeners[name] ?? []).forEach((cb) => cb(event));
  }

  emitError(dataOrNull: unknown = null) {
    const event = { data: dataOrNull == null ? undefined : JSON.stringify(dataOrNull) };
    (this.listeners["error"] ?? []).forEach((cb) => cb(event));
  }
}

vi.stubGlobal("EventSource", MockEventSource as unknown as typeof EventSource);

afterEach(() => {
  cleanup();
  MockEventSource.reset();
});
