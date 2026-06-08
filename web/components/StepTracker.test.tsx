import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import StepTracker from "@/components/StepTracker";
import type { ProgramStep, StepResult } from "@/lib/types";

// StepTracker renders one button per program step. A step is "arrived" once i < steps.length;
// arrived buttons are enabled and fire onSelect, not-arrived buttons are disabled. The wrapper
// is a progressbar carrying aria-valuenow = current + 1.

const program: ProgramStep[] = [
  { id: "a", op: "sample_frames", args: {} },
  { id: "b", op: "detect", args: {} },
  { id: "c", op: "count", args: {} },
];

function st(partial: Partial<StepResult>): StepResult {
  return { id: "x", op: "x", status: "done", source: "live", ...partial };
}

describe("StepTracker", () => {
  it("enables arrived buttons and disables not-yet-arrived ones", () => {
    // 2 steps arrived (a, b); c has not arrived.
    const steps = [st({ id: "a" }), st({ id: "b" })];
    render(<StepTracker program={program} steps={steps} current={1} onSelect={() => {}} />);
    const buttons = screen.getAllByRole("button");
    expect(buttons[0]).toBeEnabled();
    expect(buttons[1]).toBeEnabled();
    expect(buttons[2]).toBeDisabled();
  });

  it("fires onSelect only for arrived steps", () => {
    const onSelect = vi.fn();
    const steps = [st({ id: "a" }), st({ id: "b" })];
    render(<StepTracker program={program} steps={steps} current={1} onSelect={onSelect} />);
    const buttons = screen.getAllByRole("button");
    fireEvent.click(buttons[0]); // arrived -> fires
    fireEvent.click(buttons[2]); // not arrived (disabled) -> no fire
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect).toHaveBeenCalledWith(0);
  });

  it("exposes a progressbar with aria-valuenow = current + 1", () => {
    const steps = [st({ id: "a" }), st({ id: "b" }), st({ id: "c" })];
    render(<StepTracker program={program} steps={steps} current={2} onSelect={() => {}} />);
    const bar = screen.getByRole("progressbar");
    expect(bar).toHaveAttribute("aria-valuenow", "3");
    expect(bar).toHaveAttribute("aria-valuemin", "1");
    expect(bar).toHaveAttribute("aria-valuemax", "3");
  });
});
