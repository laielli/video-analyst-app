import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import Findings from "./Findings";
import type { Findings as F } from "@/lib/types";

const grounded: F = { answer: "Yes", verdict: "Yes — #10 scored the first goal", grounded: true, reason: null };
const ungrounded: F = { answer: null, verdict: null, grounded: false, reason: "codegen-disabled" };

describe("Findings", () => {
  it("ungrounded wins over resolved on a done run (precedence)", () => {
    const { container } = render(<Findings status="done" findings={ungrounded} stepCount={0} total={0} />);
    // shows the couldn't-ground copy + the REASON_COPY for codegen-disabled
    expect(screen.getByText(/Couldn.t ground an answer/i)).toBeInTheDocument();
    expect(screen.getByText(/Program generation is offline/i)).toBeInTheDocument();
    // the ungrounded class is applied (not "resolved").
    expect(container.querySelector(".findings.ungrounded")).toBeTruthy();
    expect(container.querySelector(".findings.resolved")).toBeNull();
    // the green check is NOT rendered (verdict path didn't win); verdict text absent.
    expect(screen.queryByText(grounded.verdict!)).toBeNull();
  });

  it("resolved shows the verdict on a grounded done run", () => {
    const { container } = render(<Findings status="done" findings={grounded} stepCount={7} total={7} />);
    expect(screen.getByText(grounded.verdict!)).toBeInTheDocument();
    expect(container.querySelector(".findings.resolved")).toBeTruthy();
  });

  it("running shows 'Running… step X / Y'", () => {
    render(<Findings status="running" findings={null} stepCount={3} total={7} />);
    expect(screen.getByText(/Running… step 3 \/ 7/)).toBeInTheDocument();
  });

  it("running clamps step count to total", () => {
    render(<Findings status="running" findings={null} stepCount={9} total={7} />);
    expect(screen.getByText(/Running… step 7 \/ 7/)).toBeInTheDocument();
  });

  it("idle shows 'Ready to run'; error shows 'Run failed'", () => {
    const { rerender } = render(<Findings status="idle" findings={null} stepCount={0} total={0} />);
    expect(screen.getByText("Ready to run")).toBeInTheDocument();
    rerender(<Findings status="error" findings={null} stepCount={0} total={0} />);
    expect(screen.getByText("Run failed")).toBeInTheDocument();
  });

  it("falls back to a generic reason copy for an unknown reason enum", () => {
    const weird: F = { answer: null, verdict: null, grounded: false, reason: "brand-new-reason" };
    const { container } = render(<Findings status="done" findings={weird} stepCount={0} total={0} />);
    // the ungrounded branch still wins; the unknown reason falls back to the generic ".reason" copy.
    expect(container.querySelector(".findings.ungrounded")).toBeTruthy();
    const fallback = container.querySelector(".ug-reason");
    expect(fallback?.textContent).toContain("Couldn't ground an answer.");
  });
});
