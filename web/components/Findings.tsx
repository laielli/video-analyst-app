import type { Findings as F } from "@/lib/types";
import type { RunStatus } from "@/lib/useRun";

const Check = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6 9 17l-5-5" /></svg>
);
const Spinner = () => (
  <svg className="spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M12 3a9 9 0 1 0 9 9" /></svg>
);

export default function Findings({
  status, findings, stepCount, total,
}: { status: RunStatus; findings: F | null; stepCount: number; total: number }) {
  const resolved = status === "done" && findings;
  return (
    <div className="findings-wrap">
      <span className="panel-title" style={{ color: "var(--muted)" }}>Findings</span>
      <div className={`findings ${resolved ? "resolved" : status === "running" ? "working" : ""}`} aria-live="polite">
        <span className="ic">{resolved ? <Check /> : status === "running" ? <Spinner /> : null}</span>
        <span>
          {resolved
            ? findings!.verdict
            : status === "running"
            ? `Running… step ${Math.min(stepCount, total)} / ${total || "?"}`
            : status === "error"
            ? "Run failed"
            : "Ready to run"}
        </span>
      </div>
    </div>
  );
}
