import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import ProgramPanel from "@/components/ProgramPanel";
import { commentFor } from "@/lib/program";
import type { ProgramStep, StepResult } from "@/lib/types";

// ProgramPanel classes each step block active/done/pending by `current`, shows the empty-step ○
// marker when a step's status is "empty", and renders the per-step comment from commentFor.

const program: ProgramStep[] = [
  { id: "frames", op: "sample_frames", args: { start_ms: 0, end_ms: 100, fps: 8 } },
  { id: "people", op: "detect", args: { frames: "frames", classes: ["person"] } },
  { id: "tens", op: "filter", args: { items: "people", where: { field: "text", equals: "10" } } },
];

function st(partial: Partial<StepResult>): StepResult {
  return { id: "x", op: "x", status: "done", source: "live", ...partial };
}

describe("ProgramPanel", () => {
  it("classes blocks done / active / pending around current", () => {
    const steps = [st({ id: "frames", op: "sample_frames" }), st({ id: "people", op: "detect" })];
    const { container } = render(<ProgramPanel program={program} steps={steps} current={1} />);
    const blocks = container.querySelectorAll(".step-block");
    expect(blocks[0].className).toContain("done"); // i < current
    expect(blocks[1].className).toContain("active"); // i === current
    expect(blocks[2].className).toContain("pending"); // i > current
  });

  it("shows the ○ marker for an empty completed step", () => {
    // step 0 is done-with-empty-status, current is past it.
    const steps = [st({ id: "frames", op: "sample_frames", status: "empty" }), st({ id: "people", op: "detect" })];
    const { container } = render(<ProgramPanel program={program} steps={steps} current={1} />);
    const firstMarker = container.querySelectorAll(".step-block .mk")[0];
    expect(firstMarker.textContent).toBe("○");
  });

  it("derives each comment from commentFor, with a `# <op>` fallback for an unknown op", () => {
    const withUnknown: ProgramStep[] = [...program, { id: "mystery", op: "no_such_op", args: {} }];
    render(<ProgramPanel program={withUnknown} steps={[]} current={-1} />);
    // each known op renders commentFor's derived text — tied to the real function, not baked literals.
    for (const s of program) {
      expect(screen.getByText(commentFor(s))).toBeInTheDocument();
    }
    // an op absent from commentFor's switch falls back to `# <op>`.
    expect(screen.getByText("# no_such_op")).toBeInTheDocument();
  });
});
