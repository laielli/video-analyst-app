import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import ProgramPanel from "./ProgramPanel";
import type { ProgramStep, StepResult } from "@/lib/types";

const program: ProgramStep[] = [
  { id: "frames", op: "sample_frames", args: { start_ms: 0, end_ms: 100, fps: 8 } },
  { id: "people", op: "detect", args: { frames: "frames", classes: ["person"] } },
  { id: "n", op: "count", args: { items: "people" } },
];

function steps(...statuses: StepResult["status"][]): StepResult[] {
  return statuses.map((status, i) => ({ id: program[i].id, op: program[i].op, status, source: "live" }));
}

describe("ProgramPanel", () => {
  it("classes blocks active/done/pending by current index", () => {
    const { container } = render(<ProgramPanel program={program} steps={steps("done", "done")} current={1} />);
    const blocks = Array.from(container.querySelectorAll(".step-block"));
    expect(blocks[0].className).toContain("done");    // i < current
    expect(blocks[1].className).toContain("active");  // i === current
    expect(blocks[2].className).toContain("pending"); // i > current
  });

  it("renders the comment for each op from COMMENTS", () => {
    const { container } = render(<ProgramPanel program={program} steps={steps("done", "done", "done")} current={2} />);
    const comments = Array.from(container.querySelectorAll(".comment")).map((c) => c.textContent);
    expect(comments[0]).toContain("sample frames around the goal moment");
    expect(comments[1]).toContain("find every player on the pitch");
    expect(comments[2]).toContain("how many?");
  });

  it("shows the empty-step ○ marker for a completed step whose result is empty", () => {
    // step 0 is done-and-empty, current is 1, so step 0's marker renders ○ (not the check).
    const { container } = render(<ProgramPanel program={program} steps={steps("empty", "done")} current={1} />);
    const firstMarker = container.querySelectorAll(".comment .mk")[0];
    expect(firstMarker.textContent).toBe("○");
  });

  it("renders a tokenized code line per step", () => {
    const { container } = render(<ProgramPanel program={program} steps={steps("done")} current={0} />);
    const lines = container.querySelectorAll(".code-line");
    expect(lines).toHaveLength(3);
    expect(lines[0].textContent).toContain("sample_frames");
  });
});
