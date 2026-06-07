import type { Findings as F, ProgramSource } from "@/lib/types";
import type { RunStatus } from "@/lib/useRun";
import { isUngrounded } from "@/lib/useRun";

const Check = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6 9 17l-5-5" /></svg>
);
const Spinner = () => (
  <svg className="spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M12 3a9 9 0 1 0 9 9" /></svg>
);
// A distinct mark for the honest "couldn't ground this" state — NOT the green resolved check.
const Question = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="9" /><path d="M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.7.4-1 .9-1 1.7" /><path d="M12 17h.01" /></svg>
);

// Human-readable copy for each fixed reason enum the backend emits (Q1 -> A).
const REASONS: Record<string, string> = {
  "codegen-disabled": "The program-generation model isn't configured, so this question couldn't be compiled.",
  "codegen-error": "The program-generation model couldn't compile this question.",
  "invalid-program": "The generated program didn't pass validation, so it was not executed.",
  "execution-error": "The generated program failed to execute over this clip's evidence.",
  "no-grounded-answer": "The program ran, but the clip's evidence didn't ground an answer to this question.",
  "budget-exceeded": "The per-session program-generation budget is exhausted. Try again later.",
};

export default function Findings({
  status, findings, stepCount, total, programSource,
}: {
  status: RunStatus;
  findings: F | null;
  stepCount: number;
  total: number;
  programSource?: ProgramSource;
}) {
  // Check the ungrounded branch FIRST: an ungrounded run still emits `done`, so without this it
  // would render the green resolved verdict (Phase 4 BLOCKER).
  const ungrounded = status === "done" && isUngrounded(findings);
  const resolved = status === "done" && findings && !ungrounded;

  const reasonCopy = findings?.reason ? REASONS[findings.reason] ?? findings.reason : null;

  return (
    <div className="findings-wrap">
      <span className="panel-title" style={{ color: "var(--muted)" }}>Findings</span>
      <div
        className={`findings ${ungrounded ? "ungrounded" : resolved ? "resolved" : status === "running" ? "working" : ""}`}
        aria-live="polite"
      >
        <span className="ic">
          {ungrounded ? <Question /> : resolved ? <Check /> : status === "running" ? <Spinner /> : null}
        </span>
        <span>
          {ungrounded ? (
            <>
              Couldn&apos;t ground an answer
              {reasonCopy && <span className="findings-reason"> — {reasonCopy}</span>}
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
      {programSource && (status === "done" || status === "running") && (
        <span className={`prov-tag ${programSource}`} title="Provenance of the visual program">
          {programSource}
        </span>
      )}
    </div>
  );
}
