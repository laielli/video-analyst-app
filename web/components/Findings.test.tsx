import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import Findings from "@/components/Findings";
import type { Findings as F } from "@/lib/types";

// Findings renders one of: ungrounded (couldn't ground), resolved (verdict), running, error,
// idle. The load-bearing branch is the ungrounded-BEFORE-resolved precedence: an ungrounded run
// still emits `done`, so the green Check must NOT win when grounded === false.

describe("Findings", () => {
  it("ungrounded done shows the Question mark + reason copy, NOT the verdict", () => {
    const findings: F = { answer: null, verdict: null, grounded: false, reason: "codegen-disabled" };
    render(<Findings status="done" findings={findings} stepCount={0} total={0} />);
    expect(screen.getByText(/Couldn't ground an answer/i)).toBeInTheDocument();
    // the fixed reason copy is rendered.
    expect(screen.getByText(/Program generation is offline/i)).toBeInTheDocument();
    // the ungrounded class is applied (not resolved).
    expect(document.querySelector(".findings.ungrounded")).toBeInTheDocument();
    expect(document.querySelector(".findings.resolved")).toBeNull();
  });

  it("resolved done shows the verdict", () => {
    const findings: F = { answer: "Yes", verdict: "Yes — #10 scored the first goal", grounded: true };
    render(<Findings status="done" findings={findings} stepCount={7} total={7} />);
    expect(screen.getByText("Yes — #10 scored the first goal")).toBeInTheDocument();
    expect(document.querySelector(".findings.resolved")).toBeInTheDocument();
  });

  it("running shows the step progress", () => {
    render(<Findings status="running" findings={null} stepCount={3} total={7} />);
    expect(screen.getByText("Running… step 3 / 7")).toBeInTheDocument();
    expect(document.querySelector(".findings.working")).toBeInTheDocument();
  });

  it("treats a done run with missing grounded flag as resolved (legacy/canned)", () => {
    const findings: F = { answer: "Yes", verdict: "the verdict" };
    render(<Findings status="done" findings={findings} stepCount={1} total={1} />);
    expect(screen.getByText("the verdict")).toBeInTheDocument();
    expect(document.querySelector(".findings.resolved")).toBeInTheDocument();
  });

  it("error status shows 'Run failed'", () => {
    render(<Findings status="error" findings={null} stepCount={0} total={0} />);
    expect(screen.getByText("Run failed")).toBeInTheDocument();
  });
});
