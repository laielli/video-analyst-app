import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import EvidencePanel from "@/components/EvidencePanel";
import type { StepResult } from "@/lib/types";

// EvidencePanel renders the active step's frame + overlays. The load-bearing logic is frameSrc:
// ts 0 -> no-frame empty state; ts <= 4200 -> goal still; ts > 4200 -> scorer still.

function stepWith(ts: number, overlays: StepResult["evidence"]["overlays"] = []): StepResult {
  return {
    id: "s",
    op: "detect",
    producer: "Azure AI Vision · detect",
    status: "done",
    source: "cached",
    input_label: "5 frames",
    output_label: "5 people",
    confidence: 0.75,
    evidence: { frame_ts_ms: ts, overlays },
  };
}

describe("EvidencePanel", () => {
  it("ts 0 -> 'No evidence frame' empty state, no img", () => {
    render(<EvidencePanel step={stepWith(0)} />);
    expect(screen.getByText(/No evidence frame for this step/i)).toBeInTheDocument();
    expect(document.querySelector("img")).toBeNull();
  });

  it("ts <= 4200 -> goal still", () => {
    render(<EvidencePanel step={stepWith(4000)} />);
    const img = document.querySelector("img");
    expect(img?.getAttribute("src")).toBe("/frames/goal-4000.jpg");
  });

  it("ts > 4200 -> scorer still", () => {
    render(<EvidencePanel step={stepWith(4625)} />);
    const img = document.querySelector("img");
    expect(img?.getAttribute("src")).toBe("/frames/scorer-4625.jpg");
  });

  it("renders overlays with their tone class + label", () => {
    const step = stepWith(4625, [
      { box: { x: 0.1, y: 0.2, w: 0.3, h: 0.4 }, label: "person 0.75", tone: "azure", confidence: 0.75, kind: "box" },
    ]);
    render(<EvidencePanel step={step} />);
    const ov = document.querySelector(".ov");
    expect(ov?.className).toContain("azure");
    expect(screen.getByText("person 0.75")).toBeInTheDocument();
    // confidence rendered next to the overlay label (scope to the overlay tag — 0.75 also
    // appears in the Confidence readout row, so a global getByText would be ambiguous).
    expect(ov?.querySelector(".ov-tag .conf")?.textContent?.trim()).toBe("0.75");
  });

  it("renders the program-source provenance tag", () => {
    render(<EvidencePanel step={stepWith(4625)} programSource="ungrounded" />);
    const tag = document.querySelector(".prov-tag");
    expect(tag?.className).toContain("ungrounded");
    expect(tag?.textContent).toBe("ungrounded");
  });

  it("renders the step note when present", () => {
    const step = { ...stepWith(4000), note: "no detections in the sampled frames" };
    render(<EvidencePanel step={step} />);
    expect(screen.getByText("no detections in the sampled frames")).toBeInTheDocument();
  });

  it("handles a null step gracefully (em-dash readouts)", () => {
    render(<EvidencePanel step={null} />);
    expect(screen.getByText(/No evidence frame for this step/i)).toBeInTheDocument();
  });
});
