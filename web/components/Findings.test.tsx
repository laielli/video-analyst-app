import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import Findings from "@/components/Findings";
import type { Findings as F } from "@/lib/types";

// Findings renders the verdict. The load-bearing branch is precedence: an ungrounded run still
// emits `done`, so the ungrounded check must win over the resolved (green-check) branch. Flipping
// that order in Findings.tsx should turn the first test red.

describe("Findings", () => {
  it("ungrounded-before-resolved: a done + grounded:false run shows the Question mark + reason copy, NOT the verdict", () => {
    const findings: F = { answer: null, verdict: "should not show", grounded: false, reason: "codegen-disabled" };
    const { container } = render(<Findings status="done" findings={findings} stepCount={0} total={0} />);
    // ungrounded class applied (not resolved).
    expect(container.querySelector(".findings.ungrounded")).toBeTruthy();
    expect(container.querySelector(".findings.resolved")).toBeNull();
    // the reason copy renders; the verdict does NOT.
    expect(screen.getByText(/Couldn't ground an answer/i)).toBeInTheDocument();
    expect(screen.getByText(/Program generation is offline/i)).toBeInTheDocument();
    expect(screen.queryByText("should not show")).toBeNull();
  });

  it("resolved: a done + grounded run shows the verdict", () => {
    const findings: F = { answer: "Yes", verdict: "Yes — #10 scored the first goal", grounded: true, reason: null };
    const { container } = render(<Findings status="done" findings={findings} stepCount={3} total={3} />);
    expect(container.querySelector(".findings.resolved")).toBeTruthy();
    expect(screen.getByText("Yes — #10 scored the first goal")).toBeInTheDocument();
  });

  it("running: shows 'Running… step X / Y'", () => {
    render(<Findings status="running" findings={null} stepCount={2} total={5} />);
    expect(screen.getByText("Running… step 2 / 5")).toBeInTheDocument();
  });

  it("missing reason falls back to generic ground copy", () => {
    const findings: F = { answer: null, verdict: null, grounded: false };
    render(<Findings status="done" findings={findings} stepCount={0} total={0} />);
    expect(screen.getByText(/Couldn't ground an answer/i)).toBeInTheDocument();
  });
});
