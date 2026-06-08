import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import ProgramPanel from "@/components/ProgramPanel";
import { COMMENTS } from "@/lib/program";
import type { ProgramStep, StepResult } from "@/lib/types";

const program: ProgramStep[] = [
  { id: "frames", op: "sample_frames", args: { start_ms: 3500, end_ms: 5000, fps: 8 } },
  { id: "people", op: "detect", args: { frames: "frames", classes: ["person"] } },
  { id: "tens", op: "filter", args: { items: "numbers", where: { field: "text", equals: "10" } } },
];

const doneStep = (id: string, status: StepResult["status"] = "done"): StepResult => ({
  id,
  op: "x",
  status,
  source: "live",
});

describe("ProgramPanel", () => {
  it("classes lines active/done/pending by current", () => {
    render(<ProgramPanel program={program} steps={[doneStep("frames"), doneStep("people")]} current={1} />);
    const blocks = document.querySelectorAll(".step-block");
    expect(blocks[0].className).toContain("done"); // i < current
    expect(blocks[1].className).toContain("active"); // i === current
    expect(blocks[2].className).toContain("pending"); // i > current
  });

  it("renders an empty-step '○' marker for a passed empty step", () => {
    // step 0 is empty and current is past it -> shows the ○ marker (not a check)
    render(
      <ProgramPanel
        program={program}
        steps={[doneStep("frames", "empty"), doneStep("people")]}
        current={2}
      />
    );
    const firstMarker = document.querySelector(".step-block .mk");
    expect(firstMarker?.textContent).toBe("○");
  });

  it("renders the per-op comment from COMMENTS", () => {
    render(<ProgramPanel program={program} steps={[]} current={-1} />);
    expect(screen.getByText(COMMENTS["sample_frames"])).toBeInTheDocument();
    expect(screen.getByText(COMMENTS["detect"])).toBeInTheDocument();
    expect(screen.getByText(COMMENTS["filter"])).toBeInTheDocument();
  });

  it("falls back to '# <op>' comment for an op without a COMMENTS entry", () => {
    const odd: ProgramStep[] = [{ id: "x", op: "mystery", args: {} }];
    render(<ProgramPanel program={odd} steps={[]} current={-1} />);
    expect(screen.getByText("# mystery")).toBeInTheDocument();
  });
});
