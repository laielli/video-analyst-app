import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import StepTracker from "@/components/StepTracker";
import type { ProgramStep, StepResult } from "@/lib/types";

// StepTracker renders a button per step: a step is `arrived` once it's in `steps`, and only
// arrived steps are enabled + fire onSelect. The progressbar carries aria-valuenow = current+1.

const program: ProgramStep[] = [
  { id: "frames", op: "sample_frames", args: {} },
  { id: "people", op: "detect", args: {} },
  { id: "n", op: "count", args: {} },
];

const step = (status: StepResult["status"] = "done"): StepResult => ({ id: "x", op: "x", status, source: "live" });

describe("StepTracker", () => {
  it("arrived steps are enabled, not-yet-arrived are disabled", () => {
    // only the first two steps have arrived.
    const steps = [step(), step()];
    render(<StepTracker program={program} steps={steps} current={1} onSelect={() => {}} />);
    const buttons = screen.getAllByRole("button");
    expect(buttons[0]).toBeEnabled();
    expect(buttons[1]).toBeEnabled();
    expect(buttons[2]).toBeDisabled(); // step 2 not arrived
  });

  it("onSelect fires only for arrived steps", () => {
    const onSelect = vi.fn();
    const steps = [step(), step()];
    render(<StepTracker program={program} steps={steps} current={1} onSelect={onSelect} />);
    const buttons = screen.getAllByRole("button");
    fireEvent.click(buttons[0]); // arrived -> fires with index 0
    fireEvent.click(buttons[2]); // not arrived -> no-op
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect).toHaveBeenCalledWith(0);
  });

  it("progressbar carries aria-valuenow = current + 1", () => {
    const steps = [step(), step(), step()];
    render(<StepTracker program={program} steps={steps} current={1} onSelect={() => {}} />);
    const bar = screen.getByRole("progressbar");
    expect(bar).toHaveAttribute("aria-valuenow", "2");
    expect(bar).toHaveAttribute("aria-valuemin", "1");
    expect(bar).toHaveAttribute("aria-valuemax", "3");
  });
});
