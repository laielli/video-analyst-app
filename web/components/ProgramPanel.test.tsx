import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import ProgramPanel from "@/components/ProgramPanel";
import type { ProgramStep, StepResult } from "@/lib/types";

// ProgramPanel classes each step-block active/done/pending by `current`, marks an empty step with
// the '○' glyph (not a check), and renders the per-op comment from COMMENTS.

const program: ProgramStep[] = [
  { id: "frames", op: "sample_frames", args: { start_ms: 0, end_ms: 100, fps: 8 } },
  { id: "people", op: "detect", args: { frames: "frames", classes: ["person"] } },
  { id: "n", op: "count", args: { items: "people" } },
];

const step = (status: StepResult["status"]): StepResult => ({ id: "x", op: "x", status, source: "live" });

describe("ProgramPanel", () => {
  it("classes blocks done / active / pending by current", () => {
    const steps = [step("done"), step("done"), step("done")];
    const { container } = render(<ProgramPanel program={program} steps={steps} current={1} />);
    const blocks = container.querySelectorAll(".step-block");
    expect(blocks[0].className).toContain("done");
    expect(blocks[1].className).toContain("active");
    expect(blocks[2].className).toContain("pending");
  });

  it("an empty completed step shows the ○ marker (not a check)", () => {
    const steps = [step("empty"), step("done"), step("done")];
    // current=2 so block 0 is in the done (i < current) region; status empty -> ○ marker.
    const { container } = render(<ProgramPanel program={program} steps={steps} current={2} />);
    const firstMarker = container.querySelectorAll(".step-block")[0].querySelector(".mk");
    expect(firstMarker!.textContent).toBe("○");
  });

  it("renders the per-op comment from COMMENTS", () => {
    const steps = [step("done"), step("done"), step("done")];
    const { container } = render(<ProgramPanel program={program} steps={steps} current={0} />);
    const comments = Array.from(container.querySelectorAll(".comment")).map((c) => c.textContent);
    expect(comments[0]).toContain("# sample frames around the goal moment");
    expect(comments[1]).toContain("# find every player on the pitch");
    expect(comments[2]).toContain("# how many?");
  });
});
