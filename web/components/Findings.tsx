import type { Findings as F } from "@/lib/types";
import type { RunStatus } from "@/lib/useRun";

const Check = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6 9 17l-5-5" /></svg>
);
const Spinner = () => (
  <svg className="spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M12 3a9 9 0 1 0 9 9" /></svg>
);
// Slashed-circle: the honest "couldn't ground" mark (NOT the green check).
const NoGround = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><circle cx="12" cy="12" r="9" /><path d="M5.6 5.6l12.8 12.8" /></svg>
);

// Human-readable copy for each fixed ungrounded reason (the backend never sends free text here).
const REASON_COPY: Record<string, string> = {
  "codegen-disabled": "codegen is offline (no Azure OpenAI credentials)",
  "codegen-error": "the program-generation step failed",
  "invalid-program": "the generated program didn't pass validation",
  "execution-error": "the program failed to run over this clip",
  "no-grounded-answer": "the evidence in this clip didn't ground an answer",
};

export default function Findings({
  status, findings, stepCount, total,
}: { status: RunStatus; findings: F | null; stepCount: number; total: number }) {
  // Ungrounded is checked FIRST, before the resolved/green path: an ungrounded run still emits
  // `done`, so without this order it would render a verdict with a green check (Resolved Q1 -> A).
  const ungrounded = status === "done" && findings?.grounded === false;
  const resolved = status === "done" && findings && !ungrounded;
  const cls = ungrounded ? "ungrounded" : resolved ? "resolved" : status === "running" ? "working" : "";

  return (
    <div className="findings-wrap">
      <span className="panel-title" style={{ color: "var(--muted)" }}>Findings</span>
      <div className={`findings ${cls}`} aria-live="polite">
        <span className="ic">
          {ungrounded ? <NoGround /> : resolved ? <Check /> : status === "running" ? <Spinner /> : null}
        </span>
        <span>
          {ungrounded ? (
            <>
              Couldn&rsquo;t ground an answer
              <span className="reason">
                {REASON_COPY[findings!.reason ?? ""] ?? findings!.reason ?? "no grounded evidence"}
              </span>
            </>
          ) : resolved ? (
            findings!.verdict
          ) : status === "running" ? (
            `Running… step ${Math.min(stepCount, total)} / ${total || "?"}`
          ) : status === "error" ? (
            "Run failed"
          ) : (
            "Ready to run"
          )}
        </span>
      </div>
    </div>
  );
}
