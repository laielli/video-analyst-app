import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import Findings from "@/components/Findings";
import type { Findings as F } from "@/lib/types";

// Findings renders the verdict line. The load-bearing branch is ungrounded-before-resolved:
// an ungrounded run still emits `done`, so grounded === false must NOT show the green Check.

const grounded: F = { answer: "Yes", verdict: "Yes — #10 scored the first goal", grounded: true };
const ungrounded: F = { answer: null, verdict: null, grounded: false, reason: "codegen-disabled" };

describe("Findings", () => {
  it("ungrounded-before-resolved: a done run with grounded=false shows the Question copy, not the verdict", () => {
    render(<Findings status="done" findings={ungrounded} stepCount={0} total={0} />);
    expect(screen.getByText(/Couldn.t ground an answer/i)).toBeInTheDocument();
    // The fixed-reason copy is rendered
    expect(
      screen.getByText(/Program generation is offline/i)
    ).toBeInTheDocument();
    // wrapper carries the ungrounded class (not resolved)
    const box = document.querySelector(".findings");
    expect(box?.className).toContain("ungrounded");
    expect(box?.className).not.toContain("resolved");
  });

  it("resolved: a grounded done run shows the verdict", () => {
    render(<Findings status="done" findings={grounded} stepCount={7} total={7} />);
    expect(screen.getByText("Yes — #10 scored the first goal")).toBeInTheDocument();
    const box = document.querySelector(".findings");
    expect(box?.className).toContain("resolved");
  });

  it("running: shows step X / Y progress", () => {
    render(<Findings status="running" findings={null} stepCount={3} total={7} />);
    expect(screen.getByText("Running… step 3 / 7")).toBeInTheDocument();
  });

  it("error: shows 'Run failed'", () => {
    render(<Findings status="error" findings={null} stepCount={0} total={0} />);
    expect(screen.getByText("Run failed")).toBeInTheDocument();
  });

  it("idle: shows 'Ready to run'", () => {
    render(<Findings status="idle" findings={null} stepCount={0} total={0} />);
    expect(screen.getByText("Ready to run")).toBeInTheDocument();
  });

  it("ungrounded with an unknown reason falls back to the generic copy", () => {
    const weird: F = { answer: null, verdict: null, grounded: false, reason: "made-up-reason" };
    render(<Findings status="done" findings={weird} stepCount={0} total={0} />);
    // The unknown enum falls back to the generic ground-out copy in the reason span (no crash).
    const reason = document.querySelector(".ug-reason");
    expect(reason?.textContent).toContain("Couldn't ground an answer.");
    expect(document.querySelector(".findings")?.className).toContain("ungrounded");
  });
});
