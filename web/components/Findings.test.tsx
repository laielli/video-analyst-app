import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import Findings from "./Findings";
import type { Findings as F } from "@/lib/types";

// Findings checks the ungrounded branch BEFORE the resolved branch: an ungrounded run still
// emits `done`, so the green resolved Check must NOT win when grounded === false.

const resolved: F = { answer: "Yes", verdict: "Yes — #10 scored the first goal", grounded: true };
const ungrounded: F = { answer: null, verdict: null, grounded: false, reason: "codegen-disabled", partial: true };

function svgs() {
  return document.querySelectorAll(".findings .ic svg");
}

describe("Findings", () => {
  it("ungrounded-before-resolved: a done run with grounded=false shows the Question mark + reason copy, NOT the verdict", () => {
    render(<Findings status="done" findings={ungrounded} stepCount={0} total={0} />);
    expect(screen.getByText(/Couldn.t ground an answer/i)).toBeInTheDocument();
    // the fixed reason enum is rendered as human copy.
    expect(screen.getByText(/Program generation is offline/i)).toBeInTheDocument();
    // the resolved verdict must NOT appear.
    expect(screen.queryByText("Yes — #10 scored the first goal")).not.toBeInTheDocument();
    // class reflects ungrounded, not resolved.
    expect(document.querySelector(".findings.ungrounded")).toBeInTheDocument();
    expect(document.querySelector(".findings.resolved")).not.toBeInTheDocument();
  });

  it("resolved: a done run with grounded=true shows the verdict", () => {
    render(<Findings status="done" findings={resolved} stepCount={7} total={7} />);
    expect(screen.getByText("Yes — #10 scored the first goal")).toBeInTheDocument();
    expect(document.querySelector(".findings.resolved")).toBeInTheDocument();
    expect(svgs().length).toBe(1); // a single (Check) icon
  });

  it("running: shows step X / Y progress", () => {
    render(<Findings status="running" findings={null} stepCount={3} total={7} />);
    expect(screen.getByText(/Running… step 3 \/ 7/)).toBeInTheDocument();
    expect(document.querySelector(".findings.working")).toBeInTheDocument();
  });

  it("idle: shows ready state", () => {
    render(<Findings status="idle" findings={null} stepCount={0} total={0} />);
    expect(screen.getByText("Ready to run")).toBeInTheDocument();
  });

  it("error: shows run failed", () => {
    render(<Findings status="error" findings={null} stepCount={0} total={0} />);
    expect(screen.getByText("Run failed")).toBeInTheDocument();
  });
});
