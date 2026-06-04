import type { ProgramStep, StepResult } from "@/lib/types";

const Check = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6 9 17l-5-5" /></svg>
);

export default function StepTracker({
  program, steps, current, onSelect,
}: {
  program: ProgramStep[];
  steps: StepResult[];
  current: number;
  onSelect: (i: number) => void;
}) {
  const total = program.length;
  const progress = total > 1 ? (86 * Math.max(0, current)) / (total - 1) : 0;

  return (
    <div className="track" role="progressbar" aria-valuemin={1} aria-valuemax={total} aria-valuenow={current + 1}>
      <div className="progress" style={{ width: `${progress}%` }} />
      {program.map((s, i) => {
        const arrived = i < steps.length;
        const empty = steps[i]?.status === "empty";
        const cls = i === current ? "active" : i < current ? (empty ? "empty" : "done") : "pending";
        const viewing = i === current && current < steps.length - 1 ? "viewing" : "";
        return (
          <button
            key={s.id}
            className={`stage ${cls} ${viewing}`}
            onClick={() => arrived && onSelect(i)}
            disabled={!arrived}
            aria-label={`Step ${i + 1}: ${s.op}`}
          >
            <span className="node">{i < current && !empty ? <Check /> : i + 1}</span>
            <span className="nm">{s.op}</span>
          </button>
        );
      })}
    </div>
  );
}
