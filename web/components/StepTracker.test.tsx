import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import StepTracker from "./StepTracker";
import type { ProgramStep, StepResult } from "@/lib/types";

const program: ProgramStep[] = [
  { id: "frames", op: "sample_frames", args: {} },
  { id: "people", op: "detect", args: {} },
  { id: "jerseys", op: "crop", args: {} },
];

function step(id: string, status: StepResult["status"] = "done"): StepResult {
  return { id, op: "x", status, source: "cached" };
}

describe("StepTracker — arrived / disabled buttons", () => {
  it("arrived steps are enabled; not-yet-arrived steps are disabled", () => {
    // two of three steps have arrived.
    render(<StepTracker program={program} steps={[step("frames"), step("people")]} current={1} onSelect={() => {}} />);
    const btns = screen.getAllByRole("button");
    expect(btns[0]).not.toBeDisabled(); // arrived
    expect(btns[1]).not.toBeDisabled(); // arrived
    expect(btns[2]).toBeDisabled();     // not arrived
  });

  it("onSelect fires only for arrived steps", () => {
    const onSelect = vi.fn();
    render(<StepTracker program={program} steps={[step("frames")]} current={0} onSelect={onSelect} />);
    const btns = screen.getAllByRole("button");
    fireEvent.click(btns[0]); // arrived -> fires with index 0
    fireEvent.click(btns[2]); // disabled / not arrived -> no fire
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect).toHaveBeenCalledWith(0);
  });
});

describe("StepTracker — progressbar", () => {
  it("aria-valuenow reflects current + 1, with min/max bounds", () => {
    render(<StepTracker program={program} steps={[step("frames"), step("people")]} current={1} onSelect={() => {}} />);
    const bar = screen.getByRole("progressbar");
    expect(bar).toHaveAttribute("aria-valuenow", "2"); // current(1) + 1
    expect(bar).toHaveAttribute("aria-valuemin", "1");
    expect(bar).toHaveAttribute("aria-valuemax", "3"); // total steps
  });
});
