import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import StepTracker from "./StepTracker";
import type { ProgramStep, StepResult } from "@/lib/types";

const program: ProgramStep[] = [
  { id: "frames", op: "sample_frames", args: {} },
  { id: "people", op: "detect", args: {} },
  { id: "n", op: "count", args: {} },
];

function steps(...statuses: StepResult["status"][]): StepResult[] {
  return statuses.map((status, i) => ({ id: program[i].id, op: program[i].op, status, source: "live" }));
}

describe("StepTracker", () => {
  it("arrived steps are enabled; not-yet-arrived steps are disabled", () => {
    render(<StepTracker program={program} steps={steps("done", "done")} current={1} onSelect={() => {}} />);
    const buttons = screen.getAllByRole("button");
    expect(buttons[0]).not.toBeDisabled(); // arrived
    expect(buttons[1]).not.toBeDisabled(); // arrived
    expect(buttons[2]).toBeDisabled();     // i >= steps.length -> not arrived
  });

  it("onSelect fires only for arrived steps", () => {
    const onSelect = vi.fn();
    render(<StepTracker program={program} steps={steps("done", "done")} current={1} onSelect={onSelect} />);
    const buttons = screen.getAllByRole("button");
    fireEvent.click(buttons[0]); // arrived -> fires with index 0
    fireEvent.click(buttons[2]); // not arrived -> disabled, no fire
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect).toHaveBeenCalledWith(0);
  });

  it("progressbar aria-valuenow tracks current + 1", () => {
    render(<StepTracker program={program} steps={steps("done", "done")} current={1} onSelect={() => {}} />);
    const bar = screen.getByRole("progressbar");
    expect(bar).toHaveAttribute("aria-valuenow", "2"); // current 1 -> 2
    expect(bar).toHaveAttribute("aria-valuemax", "3");
    expect(bar).toHaveAttribute("aria-valuemin", "1");
  });
});
