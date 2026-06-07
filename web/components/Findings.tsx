import type { Findings as F } from "@/lib/types";
import type { RunStatus } from "@/lib/useRun";

const Check = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6 9 17l-5-5" /></svg>
);
const Spinner = () => (
  <svg className="spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M12 3a9 9 0 1 0 9 9" /></svg>
);
const Warn = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M12 9v4" /><path d="M12 17h.01" /><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z" /></svg>
);

// Human-readable copy for the fixed ungrounded `reason` enums (never raw server text).
const REASON_COPY: Record<string, string> = {
  "codegen-disabled": "Program generation is off (no Azure OpenAI credentials configured).",
  "codegen-budget-exhausted": "The codegen call budget for this session is used up.",
  "codegen-error": "The program-generation step failed.",
  "invalid-program": "The generated program didn't pass validation, so it was never executed.",
  "execution-error": "The generated program failed while running over the clip's evidence.",
  "no-grounded-answer": "The program ran, but the evidence didn't ground an answer to this question.",
  "no-answer-step": "The program produced no answer step.",
  "answer-not-synthesizable": "The result couldn't be turned into a grounded answer.",
};

export default function Findings({
  status, findings, stepCount, total,
}: { status: RunStatus; findings: F | null; stepCount: number; total: number }) {
  // The ungrounded state still emits `done`, so check it BEFORE the resolved (green) path —
  // an honest "couldn't ground this" must NOT render as a green verdict (Q1→A, Phase 4).
  const ungrounded = status === "done" && findings && findings.grounded === false;
  const resolved = status === "done" && findings && !ungrounded;

  const cls = ungrounded ? "ungrounded" : resolved ? "resolved" : status === "running" ? "working" : "";
  const icon = ungrounded ? <Warn /> : resolved ? <Check /> : status === "running" ? <Spinner /> : null;

  let text: string;
  if (ungrounded) {
    text = "Couldn't ground an answer";
  } else if (resolved) {
    text = findings!.verdict ?? "Answered";
  } else if (status === "running") {
    text = `Running… step ${Math.min(stepCount, total)} / ${total || "?"}`;
  } else if (status === "error") {
    text = "Run failed";
  } else {
    text = "Ready to run";
  }

  const reasonCopy = ungrounded
    ? REASON_COPY[findings!.reason ?? ""] ?? "The runtime couldn't ground an answer."
    : null;

  return (
    <div className="findings-wrap">
      <span className="panel-title" style={{ color: "var(--muted)" }}>Findings</span>
      <div className={`findings ${cls}`} aria-live="polite">
        <span className="ic">{icon}</span>
        <span>{text}</span>
      </div>
      {reasonCopy && <span className="findings-reason">{reasonCopy}</span>}
    </div>
  );
}
