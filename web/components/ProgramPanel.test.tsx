import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import ProgramPanel from "./ProgramPanel";
import { COMMENTS } from "@/lib/program";
import type { ProgramStep, StepResult } from "@/lib/types";

const program: ProgramStep[] = [
  { id: "frames", op: "sample_frames", args: { start_ms: 0, end_ms: 100, fps: 8 } },
  { id: "people", op: "detect", args: { frames: "frames", classes: ["person"] } },
  { id: "jerseys", op: "crop", args: { detections: "people", region: "jersey" } },
];

function stepResult(over: Partial<StepResult>): StepResult {
  return { id: "x", op: "detect", status: "done", source: "cached", ...over };
}

function blocks() {
  return Array.from(document.querySelectorAll(".step-block"));
}

describe("ProgramPanel — classing by current", () => {
  it("classes blocks done / active / pending around current", () => {
    render(<ProgramPanel program={program} steps={[]} current={1} />);
    const b = blocks();
    expect(b[0].className).toContain("done");    // i < current
    expect(b[1].className).toContain("active");  // i === current
    expect(b[2].className).toContain("pending"); // i > current
  });

  it("renders the comment per op from COMMENTS", () => {
    render(<ProgramPanel program={program} steps={[]} current={0} />);
    expect(screen.getByText(COMMENTS["sample_frames"])).toBeInTheDocument();
    expect(screen.getByText(COMMENTS["detect"])).toBeInTheDocument();
    expect(screen.getByText(COMMENTS["crop"])).toBeInTheDocument();
  });

  it("shows the empty-step ○ marker for a done block whose step status is empty", () => {
    // step 0 is before current (current=2) AND its trace status is "empty" -> ○ marker.
    const steps = [stepResult({ id: "frames", status: "empty" }),
                   stepResult({ id: "people", status: "done" })];
    render(<ProgramPanel program={program} steps={steps} current={2} />);
    const marker = blocks()[0].querySelector(".mk");
    expect(marker?.textContent).toContain("○");
  });
});
