import type { Findings as F, UngroundedReason } from "@/lib/types";
import { isUngrounded } from "@/lib/useRun";
import type { RunStatus } from "@/lib/useRun";

const Check = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6 9 17l-5-5" /></svg>
);
const Spinner = () => (
  <svg className="spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M12 3a9 9 0 1 0 9 9" /></svg>
);
// Honest "couldn't ground" mark — NOT the green check (Q1 -> A: no fake verdict).
const Slash = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="9" /><path d="m15 9-6 6M9 9l6 6" /></svg>
);

// Map the fixed reason enum to a human-readable, non-leaky sentence.
const REASON_TEXT: Record<UngroundedReason, string> = {
  "codegen-disabled": "Codegen is offline (no Azure OpenAI credentials), so this novel question couldn't be compiled.",
  "codegen-error": "The program generator couldn't compile this question into a runnable program.",
  "invalid-program": "The generated program didn't pass validation, so it was never run (trust boundary).",
  "execution-error": "The generated program failed while running over the cached evidence.",
  "no-grounded-answer": "The program ran, but the evidence didn't ground an answer to this question.",
};

export default function Findings({
  status, findings, stepCount, total,
}: { status: RunStatus; findings: F | null; stepCount: number; total: number }) {
  // Ungrounded is checked BEFORE resolved: an ungrounded run still emits `done`, so without this
  // it would render the green check + a (null) verdict. (Phase 4 BLOCKER.)
  const ungrounded = status === "done" && isUngrounded(findings);
  const resolved = status === "done" && findings && !ungrounded;

  let body: React.ReactNode;
  if (ungrounded) {
    const reason = findings!.reason as UngroundedReason | null | undefined;
    body = (
      <span>
        Couldn&apos;t ground an answer.
        {reason && REASON_TEXT[reason] ? <span className="ungrounded-why"> {REASON_TEXT[reason]}</span> : null}
      </span>
    );
  } else if (resolved) {
    body = <span>{findings!.verdict}</span>;
  } else if (status === "running") {
    body = <span>{`Running… step ${Math.min(stepCount, total)} / ${total || "?"}`}</span>;
  } else if (status === "error") {
    body = <span>Run failed</span>;
  } else {
    body = <span>Ready to run</span>;
  }

  const cls = ungrounded ? "ungrounded" : resolved ? "resolved" : status === "running" ? "working" : "";
  return (
    <div className="findings-wrap">
      <span className="panel-title" style={{ color: "var(--muted)" }}>Findings</span>
      <div className={`findings ${cls}`} aria-live="polite">
        <span className="ic">{ungrounded ? <Slash /> : resolved ? <Check /> : status === "running" ? <Spinner /> : null}</span>
        {body}
      </div>
    </div>
  );
}
