import type { Findings as F } from "@/lib/types";
import type { RunStatus } from "@/lib/useRun";

const Check = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6 9 17l-5-5" /></svg>
);
const Spinner = () => (
  <svg className="spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M12 3a9 9 0 1 0 9 9" /></svg>
);
// Question-mark-in-circle: the honest "couldn't ground this" mark (distinct from the green Check).
const Question = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="9" /><path d="M9.5 9a2.5 2.5 0 0 1 5 0c0 1.5-2.5 2-2.5 3.5" /><path d="M12 17h.01" /></svg>
);

// Human-readable copy for each fixed ungrounded reason enum (the server never sends free text).
const REASON_COPY: Record<string, string> = {
  "codegen-disabled": "Program generation is offline (no Azure OpenAI credentials configured).",
  "codegen-error": "The program generator couldn't compile this question.",
  "invalid-program": "The generated program didn't pass validation, so it wasn't run.",
  "execution-error": "The program failed while running over this clip.",
  "no-grounded-answer": "The program ran, but the evidence in this clip didn't ground an answer.",
  "no-answer-step": "The program produced no answer step.",
  "unsupported-kind": "This question shape isn't supported by the current vision primitives.",
};

export default function Findings({
  status, findings, stepCount, total,
}: { status: RunStatus; findings: F | null; stepCount: number; total: number }) {
  // Check the ungrounded branch BEFORE the resolved branch: an ungrounded run still emits `done`,
  // so the green resolved check must NOT win. grounded === false is the honest "couldn't ground".
  const ungrounded = status === "done" && !!findings && findings.grounded === false;
  const resolved = status === "done" && !!findings && !ungrounded;

  const cls = ungrounded ? "ungrounded" : resolved ? "resolved" : status === "running" ? "working" : "";
  const reasonText = findings?.reason ? REASON_COPY[findings.reason] ?? "Couldn't ground an answer." : null;

  return (
    <div className="findings-wrap">
      <span className="panel-title" style={{ color: "var(--muted)" }}>Findings</span>
      <div className={`findings ${cls}`} aria-live="polite">
        <span className="ic">
          {ungrounded ? <Question /> : resolved ? <Check /> : status === "running" ? <Spinner /> : null}
        </span>
        <span>
          {ungrounded ? (
            <>Couldn&apos;t ground an answer{reasonText ? <span className="ug-reason"> — {reasonText}</span> : null}</>
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
