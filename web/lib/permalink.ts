// Pure helpers for the shareable-permalink affordance: build the copy-link URL from a run_id and
// parse the load-from-id / load-canned-query params off the page URL. Kept side-effect-free (no
// clipboard, no window) so they are trivially unit-testable; page.tsx wires them to the real
// navigator.clipboard + window.location.

/**
 * The shareable link for a stored run: `${origin}/?run=<run_id>`. The run_id is the backend
 * content-address, so the same question always yields the same link. Pure — no clipboard.
 */
export function buildRunLink(origin: string, runId: string): string {
  return `${origin}/?run=${encodeURIComponent(runId)}`;
}

/**
 * Parse the page's URL params we care about on load. `run` = a permalink replay (?run=<id>);
 * `query` = a canned catalog query to select (?query=<id>). At most one is meaningful; if both are
 * present `run` wins (a shared run link is the more specific intent).
 */
export function parseLoadParams(search: string): { run: string | null; query: string | null } {
  const params = new URLSearchParams(search);
  return { run: params.get("run"), query: params.get("query") };
}
