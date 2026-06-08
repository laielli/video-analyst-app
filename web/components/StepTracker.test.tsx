import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import StepTracker from "@/components/StepTracker";
import type { ProgramStep, StepResult } from "@/lib/types";

const program: ProgramStep[] = [
  { id: "frames", op: "sample_frames", args: {} },
  { id: "people", op: "detect", args: {} },
  { id: "tens", op: "filter", args: {} },
];

const step = (id: string, status: StepResult["status"] = "done"): StepResult => ({
  id,
  op: "x",
  status,
  source: "live",
});

describe("StepTracker", () => {
  it("disables stages that have not yet arrived", () => {
    // only the first step has arrived (steps.length === 1)
    render(<StepTracker program={program} steps={[step("frames")]} current={0} onSelect={() => {}} />);
    const buttons = screen.getAllByRole("button");
    expect(buttons[0]).not.toBeDisabled(); // arrived
    expect(buttons[1]).toBeDisabled(); // not arrived
    expect(buttons[2]).toBeDisabled();
  });

  it("fires onSelect only for arrived steps", () => {
    const onSelect = vi.fn();
    render(
      <StepTracker
        program={program}
        steps={[step("frames"), step("people")]}
        current={1}
        onSelect={onSelect}
      />
    );
    const buttons = screen.getAllByRole("button");
    fireEvent.click(buttons[0]); // arrived -> fires
    fireEvent.click(buttons[2]); // not arrived (disabled) -> no fire
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect).toHaveBeenCalledWith(0);
  });

  it("exposes progressbar aria-valuenow = current + 1", () => {
    render(<StepTracker program={program} steps={program.map((p) => step(p.id))} current={2} onSelect={() => {}} />);
    const bar = screen.getByRole("progressbar");
    expect(bar).toHaveAttribute("aria-valuenow", "3");
    expect(bar).toHaveAttribute("aria-valuemin", "1");
    expect(bar).toHaveAttribute("aria-valuemax", "3");
  });

  it("classes an arrived empty step as empty (not done)", () => {
    render(
      <StepTracker
        program={program}
        steps={[step("frames", "empty"), step("people")]}
        current={2}
        onSelect={() => {}}
      />
    );
    const buttons = screen.getAllByRole("button");
    expect(buttons[0].className).toContain("empty");
  });
});
